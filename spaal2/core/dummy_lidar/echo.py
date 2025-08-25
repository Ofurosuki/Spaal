class Echo:
    """
    エコーを表すクラス
    """
    """
    self.peak_position: int
        エコーのピークの位置(index)
    self.peak_height: float
        エコーのピークの高さ
    self.width: int
        エコーの幅(index)
    """
    def __init__(self, peak_position: int, peak_height: float, width: int, signal) -> None:
        self.peak_position = peak_position
        self.peak_height = peak_height
        self.width = width
        self.signal = signal