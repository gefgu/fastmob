class BaseDataFrame:
  def __init__(self, df):
    self.df = df

  def __getitem__(self, item):
    return self.df[item]

  def __getattr__(self, name):
    return getattr(self.df, name)
  
