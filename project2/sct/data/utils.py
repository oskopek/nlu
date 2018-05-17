class MissingDict(dict):
    """Replace missing values with the default value, but do not insert them."""

    def __init__(self, default_val=None, *args, **kwargs):
        super(MissingDict, self).__init__(*args, **kwargs)
        self.default_val = default_val

    def __missing__(self, key):
        return self.default_val
