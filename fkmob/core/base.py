class BaseDataFrame:
    """Small wrapper around an eager dataframe backend."""

    def __init__(self, df):
        self.df = df

    def __contains__(self, item):
        return item in self.df

    def __getitem__(self, item):
        return self.df[item]

    def __setitem__(self, key, value):
        self.df[key] = value

    def __getattr__(self, name):
        return getattr(self.df, name)

    def __iter__(self):
        return iter(self.df)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return repr(self.df)

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return self.df.columns

    def to_native(self):
        return self.df
