# -----------------------------------------------------------
# PTArray - Plaintext Tensor
# -----------------------------------------------------------
class PTArray(FHETensor[openfhe.Plaintext]):
    """Concrete tensor class for OpenFHE plaintexts."""

    def clone(self, data=None):
        return PTArray(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    @property
    def dtype(self) -> Literal["PTArray"]:
        return "PTArray"

    def decrypt(self, *args, **kwargs):
        raise NotImplementedError("Decrypt not implemented for plaintext")

    def __repr__(self) -> str:
        return f"PTArray(meta={self.meta})"

    def serialize(self) -> dict:
        raise NotImplementedError("Serialize not implemented for plaintext")

    @classmethod
    def deserialize(cls, obj: dict) -> "PTArray":
        raise NotImplementedError("Deserialize not implemented for plaintext")


def copy_tensor(tensor: "FHETensor") -> "FHETensor":
    """
    Generic copy constructor for FHETensor and subclasses.

    Parameters
    ----------
    tensor : FHETensor
        Tensor to be copied.

    Returns
    -------
    FHETensor
        A new instance with the same metadata and (optionally deep-copied) data.
    """
    import copy

    return type(tensor)(
        data=copy.deepcopy(tensor.data),
        original_shape=tensor.original_shape,
        batch_size=tensor.batch_size,
        ncols=tensor.ncols,
        order=tensor.order,
    )
