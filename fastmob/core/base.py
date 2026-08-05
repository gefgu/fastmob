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

    def compare_to(self, other, value_col, *, group_col=None, metric=None):
        """Compare ``value_col`` against ``other``, optionally grouped by ``group_col``.

        See :func:`fastmob.measures.evaluation.compare.compare_to`.
        """
        from fastmob.measures.evaluation.compare import compare_to as _compare_to

        return _compare_to(self.df, other.df, value_col, group_col=group_col, metric=metric)

    def plot_ecdf(self, value_col, *, second=None, **kwargs):
        """Build an ECDF chart for an existing numeric or list-valued column.

        Requires the optional ``fastmob[vis]`` extra.  ``second`` may be another
        Fastmob dataframe wrapper or an eager dataframe with the same column.
        """
        from fastmob.utils._common import require_optional

        vis = require_optional("fastmob.vis", "vis")
        other = getattr(second, "df", second)
        return vis.ecdf(self.df, value_col=value_col, second=other, **kwargs)

    def plot_bar(self, *, category_col, value_col, **kwargs):
        """Build a categorical bar chart from existing columns."""
        from fastmob.utils._common import require_optional

        return require_optional("fastmob.vis", "vis").bar(
            self.df, category_col=category_col, value_col=value_col, **kwargs
        )

    def plot_heatmap(self, *, x_col, y_col, value_col, **kwargs):
        """Build a long-form heatmap from existing columns."""
        from fastmob.utils._common import require_optional

        return require_optional("fastmob.vis", "vis").heatmap(
            self.df, x_col=x_col, y_col=y_col, value_col=value_col, **kwargs
        )

    def plot_scatter(self, *, x_col, y_col, **kwargs):
        """Build a scatter chart from existing columns."""
        from fastmob.utils._common import require_optional

        return require_optional("fastmob.vis", "vis").scatter(self.df, x_col=x_col, y_col=y_col, **kwargs)

    def plot_boxplot(self, *, category_col, value_col, **kwargs):
        """Build a box plot from existing long-form columns."""
        from fastmob.utils._common import require_optional

        return require_optional("fastmob.vis", "vis").boxplot(
            self.df, category_col=category_col, value_col=value_col, **kwargs
        )
