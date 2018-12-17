from collections import namedtuple
from itertools import izip

import numpy as np
from nltk import word_tokenize
from torch.nn import Module, LSTMCell
import torch
from gtd.ml.torch.utils import GPUVariable

from gtd.utils import UnicodeMixin, chunks
from gtd.ml.torch.decoder import TrainDecoder, BeamDecoder, TrainDecoderInput
from textmorph.edit_model.encoder import Encoder
from textmorph.edit_model.attention_decoder import AttentionContextCombiner
from meta_optim import OptimN2N
from encoder import EncoderOutput

from torch.autograd import Variable

class Editor(Module):
    """Editor.

    Attributes:
        encoder (Encoder)
        train_decoder (TrainDecoder)
    """

    def __init__(self, token_embedder, hidden_dim, agenda_dim, edit_dim, lamb_reg, norm_eps, norm_max, kill_edit, decoder_cell, encoder_layers, num_iter=None, eps=None, momentum=None):
        """Construct Editor.

        Args:
            token_embedder (TokenEmbedder)
            hidden_dim (int)
            agenda_dim (int)
            edit_dim (int)
            decoder_cell (DecoderCell)
            encoder_layers (int)
        """
        super(Editor, self).__init__()
        self.encoder = Encoder(token_embedder, agenda_dim, edit_dim, hidden_dim, lamb_reg, norm_eps, norm_max, kill_edit, encoder_layers,
                                   rnn_cell_factory=LSTMCell)
        context_combiner = AttentionContextCombiner()
        self.train_decoder = TrainDecoder(decoder_cell, token_embedder, context_combiner)
        self.test_decoder_beam = BeamDecoder(decoder_cell, token_embedder, context_combiner)
        update_params = list(self.train_decoder.parameters())
        self.meta_optimizer = OptimN2N(self.encoder, 
                            self.train_decoder, 
                            update_params, 
			    iters=num_iter,
                            eps=eps,
                            momentum=momentum)
        # meta_optimizer has default settings
        # Todo: add hypereparamters to tune!!!

    @classmethod
    def _batch_editor_examples(cls, examples):
        batch = lambda attr: [getattr(ex, attr) for ex in examples]
        source_words = batch('source_words')
        insert_words = batch('insert_words')
        insert_exact_words = batch('insert_exact_words')
        delete_words = batch('delete_words')
        delete_exact_words = batch('delete_exact_words')
        target_words = batch('target_words')

        edit_embed_list = [ex.edit_embed for ex in examples]

        # either they all have edit embeds, or they all don't.
        if edit_embed_list[0] is None:
            assert all(e is None for e in edit_embed_list)
            edit_embed = None
        else:
            assert all(e is not None for e in edit_embed_list)
            edit_embed = np.stack(edit_embed_list, axis=0)

        return source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, target_words, edit_embed

    def preprocess(self, examples):
        """Preprocess a batch of EditExamples, converting them into arrays.

        Args:
            examples (list[EditExample])

        Returns:
            EditorInput
        """
        source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, target_words, edit_embed = self._batch_editor_examples(
            examples)
        encoder_input = self.encoder.preprocess(source_words, insert_words, insert_exact_words, delete_words,
                                                delete_exact_words, edit_embed)
        train_decoder_input = TrainDecoderInput(target_words, self.train_decoder.word_vocab)
        return EditorInput(encoder_input, train_decoder_input)

    def forward(self, editor_input, draw_samples, draw_p=False):
        """Return the training loss.

        Args:
            editor_input (EditorInput)
            draw_samples (bool) : flag for whether to add noise for variational approx. 

        Returns:
            loss (Variable): scalar
        """
        self.encoder_output = self.encoder(editor_input.encoder_input, draw_samples, draw_p)
        #total_loss = self.train_decoder.loss(encoder_output, editor_input.train_decoder_input) <-- original.

        mean, logvar = self.encoder_output.agenda
        var_params = torch.cat([mean, logvar], 1) 
        mean_svi = Variable(self.encoder_output.agenda[0].data, requires_grad=True)
        logvar_svi = Variable(self.encoder_output.agenda[1].data, requires_grad=True)

        var_params_svi = self.meta_optimizer.forward([mean_svi, logvar_svi], self.encoder_output, editor_input.train_decoder_input)
        # encoder_output.source_embeds or  editor_input.train_decoder_input?
        # verbose False above

        self.mean_svi_final, self.logvar_svi_final = var_params_svi
        self.z_samples = self.encoder._reparameterize(self.mean_svi_final, self.logvar_svi_final)
        #self.encoder_output.agenda = self.z_samples
        self.encoder_output = EncoderOutput(self.encoder_output.source_embeds, self.encoder_output.insert_embeds, self.encoder_output.delete_embeds, self.z_samples)
        var_loss = self.train_decoder.loss(self.encoder_output, editor_input.train_decoder_input)
        var_loss.backward(retain_variables=True)
        # the above is nll_svi.
        
        var_param_grads = self.meta_optimizer.backward([self.mean_svi_final.grad, self.logvar_svi_final.grad])
        # verbose set to False above
        var_param_grads = torch.cat(var_param_grads, 1)
        var_params.backward(var_param_grads, retain_variables=True)

        #return total_loss
        return var_loss, var_params, var_param_grads

    def loss(self, examples, draw_samples=False, draw_p=False):
        """Compute loss Variable.

        Args:
            examples (list[EditExample])
            draw_samples (bool) : flag for whether to add noise for variational approx. disable at test time.

        Returns:
            loss (Variable): of shape 1
        """
        editor_input = self.preprocess(examples)
        #total_loss = self(editor_input, draw_samples, draw_p)
        var_loss, var_params, var_param_grads = self(editor_input, draw_samples, draw_p)
        #reg_loss = 0
        #if draw_samples:
        #    reg_loss += self.encoder.regularizer(editor_input.encoder_input)
        return var_loss, var_params, var_param_grads #, reg_loss

    def per_instance_losses(self, examples, draw_samples=False, batch_size=128):
        """Compute per-instance losses."""
        per_instance_loss_list = []
        for batch in chunks(examples, batch_size):
            editor_input = self.preprocess(batch)
            encoder_output = self.encoder(editor_input.encoder_input, draw_samples)
            ilosses = self.train_decoder.per_instance_losses(encoder_output, editor_input.train_decoder_input)
            per_instance_loss_list.extend([loss.data.cpu().numpy()[0] for loss in ilosses])
        return per_instance_loss_list

    def test_batch(self, examples):
        """simple batching test"""
        if len(examples) > 1:
            var_loss0, var_params0, var_param_grads0 = self.loss([examples[0]])
            lindivid = torch.abs(var_loss0) + torch.abs(torch.mean(var_params0))
            var_loss1, var_params1, var_param_grads1 = self.loss([examples[1]])
            lindivid += torch.abs(var_loss1) + torch.abs(torch.mean(var_params1))

            #lindivid = self.loss([examples[0]]) + self.loss([examples[1]])
            var_loss2, var_params2, var_param_grads2 = self.loss(examples[0:2])
            ltogether = (torch.abs(var_loss2) + torch.abs(torch.mean(var_params2))) * 2.0
            #ltogether = self.loss(examples[0:2]) * 2.0
            if abs(lindivid.data.cpu().numpy() - ltogether.data.cpu().numpy()) > 1e-5:
                print examples[0:2]
                print 'individually:'
                print lindivid
                print 'batched:'
                print ltogether
                raise Exception('batching error - examples do not produce identical results under batching')
        else:
            raise Exception('test_batch called with example list of length < 2')
        print 'Passed batching test'

    def edit(self, examples, max_seq_length=35, beam_size=5, batch_size=256):
        """Performs edits on a batch of source sentences.

        Args:
            examples (list[EditExample])
            max_seq_length (int): max # timesteps to generate for
            beam_size (int): for beam decoding
            batch_size (int): max number of examples to pass into the RNN decoder at a time.
                The total # examples decoded in parallel = batch_size / beam_size.

        Returns:
            beam_list (list[list[list[unicode]]]): a batch of beams.
            edit_traces (list[EditTrace])
        """
        beam_list = []
        edit_traces = []
        for batch in chunks(examples, batch_size / beam_size):
            beams, traces = self._edit_batch(batch, max_seq_length, beam_size)
            beam_list.extend(beams)
            edit_traces.extend(traces)
        return beam_list, edit_traces

    def _edit_batch(self, examples, max_seq_length, beam_size):
        #source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, target_words, edit_embed = self._batch_editor_examples(examples)
        #encoder_input = self.encoder.preprocess(source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, edit_embed)

        #############
        # New inference during eval,
        editor_input = self.preprocess(examples)
        self.encoder_output = self.encoder(editor_input.encoder_input)
        mean, logvar = self.encoder_output.agenda
        var_params = torch.cat([mean, logvar], 1) 
        mean_svi = Variable(self.encoder_output.agenda[0].data, requires_grad=True)
        logvar_svi = Variable(self.encoder_output.agenda[1].data, requires_grad=True)
        var_params_svi = self.meta_optimizer.forward([mean_svi, logvar_svi], self.encoder_output, editor_input.train_decoder_input)

        self.mean_svi_final, self.logvar_svi_final = var_params_svi
        self.z_samples = self.encoder._reparameterize(self.mean_svi_final, self.logvar_svi_final)
        #self.encoder_output.agenda = self.z_samples
        encoder_output = EncoderOutput(self.encoder_output.source_embeds, self.encoder_output.insert_embeds, self.encoder_output.delete_embeds, self.z_samples)
        #############
        #encoder_output = self.encoder(encoder_input)
        beams, decoder_traces = self.test_decoder_beam.decode(examples, 
            encoder_output, 
            weighted_value_estimators=[], 
            beam_size=beam_size, 
            prefix_hints = [[]], 
            sibling_penalty=0, 
            max_seq_length=max_seq_length)

        return beams, [EditTrace(ex, d_trace.beam_traces[-1]) for ex, d_trace in izip(examples, decoder_traces)]

    def interact(self, beam_size=8, verbose=True):
        ex = EditExample.from_prompt()
        output_words_batch, edit_traces = self.edit([ex], beam_size=beam_size)
        output_words = output_words_batch[0]
        edit_trace = edit_traces[0]

        # nll = lambda example: self.loss([example]).data[0]

        # TODO: make this fully generative in the right way.. current NLL is wrong, disabled for now.
        # compare NLL of correct output and predicted output
        # output_ex = EditExample(ex.source_words, ex.insert_words, ex.delete_words, output_words)
        # gold_nll = nll(ex)
        # output_nll = nll(output_ex)

        print 'output:'
        print ' '.join(output_words)

        if verbose:
            # print
            # print 'output NLL: {}, gold NLL: {}'.format(output_nll, gold_nll)
            print edit_trace


class EditTrace(UnicodeMixin):
    def __init__(self, example, decoder_trace):
        """

        Args:
            example (EditExample)
            decoder_trace (DecoderTrace)
        """
        self.example = example
        self.decoder_trace = decoder_trace

    def __unicode__(self):
        return u'\n'.join([unicode(self.example), unicode(self.decoder_trace)])


class EditExample(namedtuple('EditExample', ['source_words', 'insert_words', 'insert_exact_words', 'delete_words',
                                             'delete_exact_words', 'target_words', 'edit_embed'])):
    """An example of how to perform an edit.

    Attributes:
        source_words (list[unicode]): a source sequence
        insert_words (list[unicode]): a list of insert words
        insert_exact_words (list[unicode]): a list of insert words to be executed exactly
        delete_words (list[unicode]): a list of delete words
        delete_exact_words (list[unicode]): a list of delete words to be executed excactly.
        target_words (list[unicode]): a target sequence. Can be None.
        edit_embed (np.ndarray): a 1D array of shape [edit_dim]. Can be None.
    """
    __slots__ = ()

    @classmethod
    def from_prompt(cls):
        get_input = lambda prompt: word_tokenize(raw_input(prompt).decode('utf-8'))
        source_words = get_input('Enter a source sentence:\n')
        whitelist = sorted(get_input('Enter whitelist (OK to leave empty):\n'))
        blacklist = sorted(get_input('Enter blacklist (OK to leave empty):\n'))
        target_words = get_input('Enter target sentence (OK to leave empty):\n')

        return EditExample(source_words, [], whitelist, [], blacklist, target_words)

    @classmethod
    def whitelist_blacklist(cls, src_words, trg_words):
        """Use the whitelist/blacklist strategy to compute edit sets.

        Since this function is called at data-loading time, all words are assumed exact.

        Args:
            src_words (list[unicode])
            trg_words (list[unicode])

        Returns:
            EditExample
        """
        src_set, trg_set = set(src_words), set(trg_words)
        whitelist = sorted(trg_set)  # all target words
        blacklist = sorted(src_set - trg_set)  # words that only appear in the source
        return EditExample(source_words=src_words, insert_words=[], insert_exact_words=whitelist,
                           delete_words=[], delete_exact_words=blacklist, target_words=trg_words)

    @classmethod
    def salient_diff(cls, src_words, trg_words, free_set):
        """Take the diff of the source and target, excluding specified "free" words.

        Args:
            src_words (list[unicode])
            trg_words (list[unicode])
            free_set (set[unicode])

        Returns:
            EditExample
        """
        src_set, trg_set = set(src_words), set(trg_words)
        insert_words = sorted(trg_set - src_set - free_set)
        delete_words = sorted(src_set - trg_set - free_set)

        return EditExample(source_words=src_words, insert_words=[], insert_exact_words=insert_words,
                           delete_words=[], delete_exact_words=delete_words, target_words=trg_words)

    def __new__(cls, source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, target_words,
                edit_embed=None):
        if edit_embed is not None:
            assert len(edit_embed.shape) == 1  # must be 1D array
        self = super(EditExample, cls).__new__(cls, source_words, insert_words, insert_exact_words, delete_words,
                                               delete_exact_words, target_words, edit_embed)
        return self

    def __repr__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        fmt = lambda name, delim, tokens: u'{}: {}'.format(name, delim.join(tokens))
        insert = self.insert_words
        delete = self.delete_words
        insert_exact = self.insert_exact_words
        delete_exact = self.delete_exact_words

        lines = []
        lines.append(fmt(u'SOURCE', u' ', self.source_words))
        if insert:
            lines.append(u'INSERT: {}'.format(insert))
        if delete:
            lines.append(u'DELETE: {}'.format(delete))
        if insert_exact:
            lines.append(u'INSERT EXACT: {}'.format(insert_exact))
        if delete_exact:
            lines.append(u'DELETE EXACT: {}'.format(delete_exact))
        if self.target_words:
            lines.append(fmt(u'TARGET', u' ', self.target_words))
        return '\n'.join(lines)


EditorInput = namedtuple('EditorInput', ['encoder_input', 'train_decoder_input'])
"""A preprocessed EditExample.

Attributes:
    encoder_input (EncoderInput)
    train_decoder_input (TrainDecoderInput)
"""
