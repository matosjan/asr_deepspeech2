import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    list_audio = []
    list_spgram = []
    list_text = []
    list_text_enc = []
    list_path = []
    list_spgram_len = []
    list_text_enc_len = []

    for elem in dataset_items:
        list_audio.append(elem["audio"].t())
        list_spgram.append(elem["spectrogram"].transpose(0, 2))
        list_text.append(elem["text"])
        list_text_enc.append(elem["text_encoded"].t())
        list_path.append(elem["audio_path"])

        list_spgram_len.append(elem["spectrogram"].shape[2])
        list_text_enc_len.append(elem["text_encoded"].shape[1])

    batch_audio = pad_sequence(list_audio, batch_first=True).squeeze()
    batch_spgram = pad_sequence(list_spgram, batch_first=True).squeeze(-1).transpose(1, 2)
    batch_text = list_text
    batch_text_enc = pad_sequence(list_text_enc, batch_first=True).squeeze()
    batch_path = list_path
    batch_spgram_len = torch.tensor(list_spgram_len)
    batch_text_enc_len = torch.tensor(list_text_enc_len)

    result_batch = {
        "audio": batch_audio,
        "spectrogram": batch_spgram,
        "text": batch_text,
        "text_encoded": batch_text_enc,
        "audio_path": batch_path,
        "spectrogram_length": batch_spgram_len,
        "text_encoded_length": batch_text_enc_len,
    }

    return result_batch
