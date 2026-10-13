from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch
import os
import numpy as np
import json
import onnxruntime as ort
import tqdm
from huggingface_hub import snapshot_download

class IndicASRConfig(PretrainedConfig):
    model_type = "iasr"
    
    def __init__(self, ts_folder: str = "path", BLANK_ID: int = 256, RNNT_MAX_SYMBOLS: int = 10,
                 PRED_RNN_LAYERS: int = 2, PRED_RNN_HIDDEN_DIM: int = 640, SOS: int = 5632, **kwargs):
        super().__init__(**kwargs)
        self.ts_folder = ts_folder
        self.BLANK_ID = BLANK_ID
        self.RNNT_MAX_SYMBOLS = RNNT_MAX_SYMBOLS
        self.PRED_RNN_LAYERS = PRED_RNN_LAYERS
        self.PRED_RNN_HIDDEN_DIM = PRED_RNN_HIDDEN_DIM
        self.SOS = SOS
        if 'FRAME_DURATION_MS' not in kwargs:
            print('Please check FRAME_DURATION_MS. The timestamps can be inaccurate')
            fs = 0.04
        else:
            fs = kwargs['FRAME_DURATION_MS']
        self.FRAME_DURATION_MS = fs

class IndicASRModel(PreTrainedModel):
    config_class = IndicASRConfig

    def __init__(self, config):
        super().__init__(config)
        
        # Load model components
        self.models = {}
        names = ['encoder', 'ctc_decoder', 'rnnt_decoder_embed', 'rnnt_decoder_rnn', 'joint_enc', 'joint_pred', 'joint_pre_net'] + [f'joint_post_net_{z}' for z in ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']]
        self.models = {}
        self.d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models['preprocessor'] = torch.jit.load(f'{config.ts_folder}/assets/preprocessor.ts', map_location=self.d)
        for n in tqdm.tqdm(names):
            component_name = f'{config.ts_folder}/assets/{n}.onnx'
            if os.path.exists(config.ts_folder):
                self.models[n] = ort.InferenceSession(component_name, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])
            else:
                self.models[n] = None   
                print('Failed to load', component_name)

        # Load vocab and language masks
        with open(f'{config.ts_folder}/assets/vocab.json') as reader:
            self.vocab = json.load(reader)
        
        with open(f'{config.ts_folder}/assets/language_masks.json') as reader:
            self.language_masks = json.load(reader)
    
    def forward(self, wav, lang, decoding='ctc', compute_timestamps=None):
        encoder_outputs, encoded_lengths = self.encode(wav)
        if decoding == 'ctc':
            return self._ctc_decode(encoder_outputs, encoded_lengths, lang, compute_timestamps)
        if decoding == 'rnnt':
            return self._rnnt_decode(encoder_outputs, encoded_lengths, lang)

    def batched_forward(self, wavs, lengths, lang, decoding='rnnt'):
        encoder_outputs, encoded_lengths = self.batched_encode(wavs, lengths)
        if decoding == 'ctc':
            raise NotImplementedError("CTC decoding is not implemented for batched forward")
        if decoding == 'rnnt':
            return self._rnnt_decode_batch(encoder_outputs, encoded_lengths, lang)

    def batched_encode(self, wavs, lengths):
        audio_signal, length = self.models['preprocessor'](input_signal=wavs.to(self.d), length=lengths.to(self.d))
        outputs, encoded_lengths = self.models['encoder'].run(['outputs', 'encoded_lengths'], {'audio_signal': audio_signal.cpu().numpy(), 'length': length.cpu().numpy()})
        return outputs, encoded_lengths

    def encode(self, wav):
        # pass through preprocessor
        audio_signal, length = self.models['preprocessor'](input_signal=wav.to(self.d), length=torch.tensor([wav.shape[-1]]).to(self.d))
        outputs, encoded_lengths = self.models['encoder'].run(['outputs', 'encoded_lengths'], {'audio_signal': audio_signal.cpu().numpy(), 'length': length.cpu().numpy()})
        return outputs, encoded_lengths
    
    def compute_timestamps(self, batch_logprobs, lens, lang, _type='w'):
        """
        Return a list of lists — one (token, t0, t1) tuple per contiguous token.
        """
        assert _type in ['w','c']
        results = []
        results_word = []
        for b in range(batch_logprobs.size(0)):
            T        = lens[b].item()                     # encoder frames for sample b
            lp       = batch_logprobs[b, :T]              # (T, vocab)
            path     = lp.argmax(dim=-1).cpu()            # greedy CTC path
            step_sec = self.config.FRAME_DURATION_MS                        # seconds per encoder frame

            segments, cur_tok, start_f = [], None, 0
            segments_word = []
            for f, tok in enumerate(path):
                tok = tok.item()
                if tok == self.config.BLANK_ID:
                    if cur_tok is not None:               # flush current segment
                        segments.append(
                            (self.vocab[lang][cur_tok], start_f*step_sec, f*step_sec)
                        )
                        cur_tok = None
                elif tok != cur_tok:                      # token changed
                    if cur_tok is not None:
                        segments.append(
                            (self.vocab[lang][cur_tok], start_f*step_sec, f*step_sec)
                        )
                    cur_tok, start_f = tok, f

            if cur_tok is not None:                       # flush tail
                segments.append(
                    (self.vocab[lang][cur_tok], start_f*step_sec, T*step_sec)
                )

            if _type == 'w':
                word = ''
                start_t = None

                for token, t0, t1 in segments:  # assuming batch size = 1
                    if '▁' in token:
                        if word:
                            # save the previous word
                            segments_word.append((word, start_t, prev_t1))
                        # start new word
                        word = token.replace('▁', '')
                        start_t = t0
                    else:
                        word += token

                    prev_t1 = t1  # track end of current token

                # flush the last word if any
                if word:
                    segments_word.append((word, start_t, prev_t1))

            results.append(segments)
            results_word.append(segments_word)
        return results if _type == 'c' else results_word

    def _ctc_decode(self, encoder_outputs, encoded_lengths, lang, compute_timestamps=None):
        logprobs = self.models['ctc_decoder'].run(['logprobs'], {'encoder_output': encoder_outputs})[0]
        logprobs = torch.from_numpy(logprobs[:, :, self.language_masks[lang]]).log_softmax(dim=-1)

        # currently no batching
        indices = torch.argmax(logprobs[0],dim=-1)
        collapsed_indices = torch.unique_consecutive(indices, dim=-1)

        hyp = ''.join([self.vocab[lang][x] for x in collapsed_indices if x != self.config.BLANK_ID]).replace('▁',' ').strip()
        
        if compute_timestamps:
            return hyp, self.compute_timestamps(logprobs, encoded_lengths, lang, _type=compute_timestamps)
        else:
            del logprobs, indices, collapsed_indices
            return hyp 
    
    def _rnnt_decode(self, encoder_outputs, encoded_lengths, lang):    
        joint_enc = self.models['joint_enc'].run(['output'], {'input': encoder_outputs.transpose(0, 2, 1)})[0]
        joint_enc = torch.from_numpy(joint_enc)
        # Initialize hypothesis with SOS token
        hyp = [self.config.SOS]
        prev_dec_state = (np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32),
                          np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32))
        
        # Iterate over time steps (T)
        for t in range(joint_enc.size(1)):
            f = joint_enc[:, t, :].unsqueeze(1)  # B x 1 x H

            not_blank = True
            symbols_added = 0

            while not_blank and ((self.config.RNNT_MAX_SYMBOLS is None) or (symbols_added < self.config.RNNT_MAX_SYMBOLS)):
                # Decoder forward passsaa
                g = self.models['rnnt_decoder_embed'].run(['output'], {'input': np.array([[hyp[-1]]], dtype=np.int64)})[0]
                g, dec_state_0, dec_state_1 = self.models['rnnt_decoder_rnn'].run(
                    ['output', 'h_o', 'c_o'],
                    {'input': g,
                     'h_i': prev_dec_state[0],
                     'c_i': prev_dec_state[1]})
                
                # Joint network
                # g = self.models['joint_pred'].run(['output'], {'input': g.transpose(0,2,1)})[0]
                g = self.models['joint_pred'].run(['output'], {'input': g})[0]

                joint_out = f + g  # B x 1 x H
                joint_out = self.models['joint_pre_net'].run(['output'], {'input': joint_out.numpy()})[0]
            
                logits = self.models[f'joint_post_net_{lang}'].run(['output'], {'input': joint_out})[0]
                log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                pred_token = log_probs.argmax(dim=-1).item()

                # Append if not blank
                if pred_token == self.config.BLANK_ID:
                    not_blank = False
                else:
                    prev_dec_state = (dec_state_0, dec_state_1)
                    hyp.append(pred_token)

                symbols_added += 1  
        
        pred_text = ''.join([self.vocab[lang][x] for x in hyp if x != self.config.SOS]).replace('▁',' ').strip()
        del joint_enc, prev_dec_state, hyp
        return pred_text

    def _rnnt_decode_batch(self, encoder_outputs, encoded_lengths, lang):
        """
        Batched greedy RNNT decoding.
        Args:
            encoder_outputs: Tensor (B x T x H)
            encoded_lengths: List[int] (lengths of each sample)
            lang: str (language id for vocab)
        Returns:
            List[str] decoded hypotheses
        """
        # Normalize encoded lengths to a Python list of ints (one per batch item)
        enc_lens = np.array(encoded_lengths).squeeze()
        if enc_lens.ndim == 0:
            enc_lens = np.array([int(enc_lens)])

        # Joint encoder forward (batch all)
        joint_enc = self.models['joint_enc'].run(
            ['output'], {'input': encoder_outputs.transpose(0, 2, 1)}
        )[0]
        joint_enc = torch.from_numpy(joint_enc)  # (B x T x H)
        
        B, T, H = joint_enc.shape
        
        # Initialize hypotheses and decoder states
        hyps = [[self.config.SOS] for _ in range(B)]
        prev_dec_states = [
            (np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32),
            np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32))
            for _ in range(B)
        ]
        # Iterate over time steps
        for t in range(T):

            f = joint_enc[:, t, :].unsqueeze(1)  # (B x 1 x H)

            # While loop for each sample (batch-wise)
            symbols_added = [0] * B
            # Only process samples whose encoded length > current timestep
            active = enc_lens >= t #torch.tensor([t < enc_lens[i] for i in range(B)], dtype=torch.bool)
            not_blank = active.copy()

            while not_blank.any() and (
                self.config.RNNT_MAX_SYMBOLS is None or 
                any(s < self.config.RNNT_MAX_SYMBOLS for s in symbols_added)
            ):
                batch_inputs = []
                batch_states_0 = []
                batch_states_1 = []
                batch_idx = []

                # Collect only active samples
                for i in range(B):
                    if bool(not_blank[i]) and (
                        self.config.RNNT_MAX_SYMBOLS is None or
                        symbols_added[i] < self.config.RNNT_MAX_SYMBOLS
                    ):
                        batch_inputs.append([hyps[i][-1]])
                        batch_states_0.append(prev_dec_states[i][0])
                        batch_states_1.append(prev_dec_states[i][1])
                        batch_idx.append(i)

                if not batch_inputs:
                    break

                # Decoder forward
                g_0 = self.models['rnnt_decoder_embed'].run(['output'], {'input': np.array(batch_inputs, dtype=np.int64)})[0].transpose(1,0,2)

                # breakpoint()
                g, dec_state_0, dec_state_1 = self.models['rnnt_decoder_rnn'].run(
                    ['output', 'h_o', 'c_o'],
                    {
                        'input': g_0,
                        'h_i': np.concatenate(batch_states_0, axis=1),  # L x B x H
                        'c_i': np.concatenate(batch_states_1, axis=1)
                    }
                )

                # Joint prediction
                g = self.models['joint_pred'].run(['output'], {'input': g})[0].transpose(1,0,2)

                # Broadcast encoder f for these samples
                f_sel = f[batch_idx]
                joint_out = f_sel + torch.from_numpy(g)  # (B_active x 1 x H)

                joint_out = self.models['joint_pre_net'].run(
                    ['output'], {'input': joint_out.numpy()}
                )[0]

                logits = self.models[f'joint_post_net_{lang}'].run(
                    ['output'], {'input': joint_out}
                )[0]

                log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                pred_tokens = log_probs.argmax(dim=-1).squeeze(1).tolist()

                # Update hyps & states
                for j, i in enumerate(batch_idx):
                    if pred_tokens[j] == self.config.BLANK_ID:
                        not_blank[i] = False
                    else:
                        hyps[i].append(pred_tokens[j])
                        prev_dec_states[i] = (
                            dec_state_0[:, j:j+1, :], 
                            dec_state_1[:, j:j+1, :]
                        )
                    symbols_added[i] += 1

        # Convert to text
        pred_texts = [
            ''.join([self.vocab[lang][x] for x in hyp if x != self.config.SOS]).replace('▁', ' ').strip()
            for hyp in hyps
        ] 
        del joint_enc, prev_dec_states, hyps
        return pred_texts

    @classmethod
    def from_pretrained(cls,
        pretrained_model_name_or_path,
        *,
        force_download=False,
        resume_download=None,
        proxies=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        revision=None, **kwargs):
        loc = snapshot_download(repo_id=pretrained_model_name_or_path, token=token)
        return cls(IndicASRConfig(ts_folder=loc, **kwargs))


if __name__ == '__main__':
    from transformers import AutoConfig, AutoModel

    # # Register the model so it can be used with AutoModel
    AutoConfig.register("iasr", IndicASRConfig)
    AutoModel.register(IndicASRConfig, IndicASRModel)