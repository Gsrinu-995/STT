from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch
import os
import json
from huggingface_hub import snapshot_download

class IndicASRConfig(PretrainedConfig):
    model_type = "iasr"
    
    def __init__(self, ts_folder: str = "path", BLANK_ID: int = 256, RNNT_MAX_SYMBOLS: int = 10,
                 PRED_RNN_LAYERS: int = 2, PRED_RNN_HIDDEN_DIM: int = 640, SOS: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.ts_folder = ts_folder
        self.BLANK_ID = BLANK_ID
        self.RNNT_MAX_SYMBOLS = RNNT_MAX_SYMBOLS
        self.PRED_RNN_LAYERS = PRED_RNN_LAYERS
        self.PRED_RNN_HIDDEN_DIM = PRED_RNN_HIDDEN_DIM
        self.SOS = SOS

class IndicASRModel(PreTrainedModel):
    config_class = IndicASRConfig

    def __init__(self, config):
        super().__init__(config)
        
        # Load model components
        self.models = {}
        names = ['preprocessor','encoder', 'ctc_decoder', 'rnnt_decoder', 'joint_enc', 'joint_pred', 'joint_pre_net'] + \
                [f'joint_post_net_{z}' for z in ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']]

        for n in names:
            component_name = f'{config.ts_folder}/assets/{n}.ts'
            if os.path.exists(component_name):
                self.models[n] = torch.jit.load(component_name, map_location = torch.device(self.config.device))
            else:
                self.models[n] = None   
                print(f'Failed to load {component_name}')

        # Load vocab and language masks
        with open(f'{config.ts_folder}/assets/vocab.json') as reader:
            self.vocab = json.load(reader)
        
        with open(f'{config.ts_folder}/assets/language_masks.json') as reader:
            self.language_masks = json.load(reader)
    
    def forward(self, wav, lang, decoding='ctc'):
        encoder_outputs, encoded_lengths = self.encode(wav)
        if decoding == 'ctc':
            return self._ctc_decode(encoder_outputs, encoded_lengths, lang)
        if decoding == 'rnnt':
            return self._rnnt_decode(encoder_outputs, encoded_lengths, lang)

    def encode(self, wav):
        audio_signal, length = self.models['preprocessor'](input_signal=wav.to(torch.device(self.config.device)), length=torch.tensor([wav.shape[-1]]).to(torch.device(self.config.device)))
        outputs, encoded_lengths = self.models['encoder'](audio_signal=audio_signal.to(torch.device(self.config.device)), length=length.to(torch.device(self.config.device)))
        return outputs, encoded_lengths

    def _ctc_decode(self, encoder_outputs, encoded_lengths, lang):
        logprobs = self.models['ctc_decoder'](encoder_output=encoder_outputs) 
        logprobs = logprobs[:,:,self.language_masks[lang]].log_softmax(dim=-1)
        indices = torch.argmax(logprobs[0],dim=-1)
        collapsed_indices = torch.unique_consecutive(indices, dim=-1)
        return ''.join([self.vocab[lang][x] for x in collapsed_indices if x != self.config.BLANK_ID]).replace('▁',' ').strip()
    
    def _rnnt_decode(self, encoder_outputs, encoded_lengths, lang):    
        joint_enc = self.models['joint_enc'](encoder_outputs.transpose(1, 2))
        hyp = [self.config.SOS]
        prev_dec_state = (torch.zeros(self.config.PRED_RNN_LAYERS,1,self.config.PRED_RNN_HIDDEN_DIM).to(torch.device(self.config.device)), 
                          torch.zeros(self.config.PRED_RNN_LAYERS,1,self.config.PRED_RNN_HIDDEN_DIM).to(torch.device(self.config.device)))

        for t in range(joint_enc.size(1)):
            f = joint_enc[:, t, :].unsqueeze(1)
            not_blank = True
            symbols_added = 0

            while not_blank and ((self.config.RNNT_MAX_SYMBOLS is None) or (symbols_added < self.config.RNNT_MAX_SYMBOLS)):
                g, _, dec_state = self.models['rnnt_decoder'](targets=torch.Tensor([[hyp[-1]]]).long().to(torch.device(self.config.device)), target_length=torch.tensor([1]).to(torch.device(self.config.device)), states=prev_dec_state)
                g = self.models['joint_pred'](g.transpose(1,2))
                joint_out = f + g  
                joint_out = self.models['joint_pre_net'](joint_out)
                logits = self.models[f'joint_post_net_{lang}'](joint_out)
                log_probs = logits.log_softmax(dim=-1)
                pred_token = log_probs.argmax(dim=-1).item()
                
                if pred_token == self.config.BLANK_ID:
                    not_blank = False
                else:
                    hyp.append(pred_token)
                    prev_dec_state = dec_state
                symbols_added += 1
        
        return ''.join([self.vocab[lang][x] for x in hyp if x != self.config.SOS]).replace('▁',' ').strip()
    
    def _save_pretrained(self, save_directory) -> None:
        # define how to serialize your model
        os.makedirs(f'{save_directory}/assets', exist_ok=True)
        for m_name, m in self.models.items():
            if m is not None:
                m.save(os.path.join(save_directory,'assets',m_name+'.ts'))
        
        # load the vocab
        with open(f'{save_directory}/assets/vocab.json','w') as writer:
            print(json.dumps(self.vocab),file=writer)
        
        # load the language_masks
        with open(f'{save_directory}/assets/language_masks.json','w') as writer:
            print(json.dumps(self.language_masks),file=writer)

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
