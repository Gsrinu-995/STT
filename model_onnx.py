from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch
import os
import numpy as np
import json
import onnxruntime as ort
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

        # timestamping
        if 'FRAME_DURATION_MS' not in kwargs:
            print('Please check FRAME_DURATION_MS. The timestamps can be inaccurate')
            fs = 0.08
        else:
            fs = kwargs['FRAME_DURATION_MS']
        
        self.FRAME_DURATION_MS = fs

class IndicASRModel(PreTrainedModel):
    config_class = IndicASRConfig

    def __init__(self, config):
        super().__init__(config)
        
        # Load model components
        self.models = {}
        names = ['encoder', 'ctc_decoder', 'rnnt_decoder', 'joint_enc', 'joint_pred', 'joint_pre_net'] + [f'joint_post_net_{z}' for z in ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']]
        self.models = {}
        self.d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models['preprocessor'] = torch.jit.load(f'{config.ts_folder}/assets/preprocessor.ts', map_location=self.d)
        for n in names:
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

    def encode(self, wav):
        # pass through preprocessor
        audio_signal, length = self.models['preprocessor'](input_signal=wav.to(self.d), length=torch.tensor([wav.shape[-1]]).to(self.d))
        outputs, encoded_lengths = self.models['encoder'].run(['outputs', 'encoded_lengths'], {'audio_signal': audio_signal.cpu().numpy(), 'length': length.cpu().numpy()})
        return outputs, encoded_lengths
    
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
                g, _, dec_state_0, dec_state_1 = self.models['rnnt_decoder'].run(
                    ['outputs', 'prednet_lengths', 'states', '162'],
                    {'targets': np.array([[hyp[-1]]], dtype=np.int32),
                     'target_length': np.array([1], dtype=np.int32),
                     'states.1': prev_dec_state[0],
                     'onnx::Slice_3': prev_dec_state[1]})
                
                # Joint network
                g = self.models['joint_pred'].run(['output'], {'input': g.transpose(0,2,1)})[0]

                joint_out = f + g  # B x 1 x H
                joint_out = self.models['joint_pre_net'].run(['output'], {'input': joint_out.numpy()})[0]
            
                logits = self.models[f'joint_post_net_{lang}'].run(['output'], {'input': joint_out})[0]
                log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                pred_token = log_probs.argmax(dim=-1).item()

                # Append if not blank
                if pred_token == self.config.BLANK_ID:
                    not_blank = False
                else:
                    hyp.append(pred_token)
                    prev_dec_state = (dec_state_0, dec_state_1)

                symbols_added += 1

        pred_text = ''.join([self.vocab[lang][x] for x in hyp if x != self.config.SOS]).replace('▁',' ').strip()
        return pred_text

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

    # Register the model so it can be used with AutoModel
    AutoConfig.register("iasr", IndicASRConfig)
    AutoModel.register(IndicASRConfig, IndicASRModel)

