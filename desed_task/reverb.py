import torch
import torchaudio
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

SAMPLE_RIR_PATH = f'{THIS_DIR}/res/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav'
print(SAMPLE_RIR_PATH)

#SAMPLE_WAV_SPEECH_PATH = '/Users/marcobertola/Repository/DESED_task/data/dcase/dataset/dcase_synth/audio/train/synthetic21_train/soundscapes/16.wav'

#path_audio = f"/Users/marcobertola/Downloads/save_example_audio.wav"
#path_speech = f"/Users/marcobertola/Downloads/save_example_speech.wav"
#path_augmented = f"/Users/marcobertola/Downloads/save_example_augmented.wav"
#path_multi_rir = f"/Users/marcobertola/Downloads/save_example_multi_rir.wav"


######
def _get_sample(path, resample=None):
    effects = [
        ["remix", "1"]
    ]
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate


def reverb(audio):
    sample_rate = 16000
    rir_raw, _ = get_rir_sample(resample=sample_rate)

    rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])

    tensors = []
    for i, _tensor in enumerate(audio):
        _tensor = _tensor.view(1, 160000)
        speech_ = torch.nn.functional.pad(_tensor, (rir.shape[1] - 1, 0))
        augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
        tensors.append(augmented)

    my_tensor = torch.stack(tensors, dim=1)
    my_tensor = torch.squeeze(my_tensor)
    return my_tensor


def main():
    print('STUBBED')


if __name__ == "__main__":
    main()
