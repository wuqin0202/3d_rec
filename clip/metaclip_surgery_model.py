from .clip_surgery_model import CLIPSurgery

class MetaCLIPSurgery(CLIPSurgery):
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        index = (text == 2).nonzero()
        x = x[index[:, 0], index[:, 1]] @ self.text_projection
        return x
