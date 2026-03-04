import torch


def predict(model, input_str, dataset, device, max_len=6):
    """
    Generate a reversed string prediction using the trained model.
    """

    model.eval()

    with torch.no_grad():

        # Convert input string → tensor
        src = torch.tensor(
            [dataset.char2idx[c] for c in input_str]
        ).unsqueeze(0).to(device)

        # Start decoder with <SOS>
        tgt = torch.tensor([[dataset.char2idx["<SOS>"]]]).to(device)

        for _ in range(max_len):

            output = model(src, tgt)

            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

        # Remove <SOS> token before decoding
        return dataset.decode(tgt[0][1:])