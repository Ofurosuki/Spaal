from spaal2.core.dummy_lidar.echo import Echo

class EchoGroup:
    """
    複数回発射などの場合用に、エコーの集合を表現するクラス
    """
    """
    self.echoes: list[Echo]
        エコーのリスト。出現順に整列
    """
    def __init__(self, echoes: list[Echo]) -> None:
        self.echoes = echoes

    def __getitem__(self, index: int) -> Echo:
        return self.echoes[index]
    
    def __len__(self) -> int:
        return len(self.echoes)