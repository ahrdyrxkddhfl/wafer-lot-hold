"""MES 규칙 위반 오류."""


class RuleViolation(Exception):
    """요청이 MES 규칙에 어긋나 거절됐다.

    rule은 README "막는 상태" 표의 번호(I1~I13)이거나 아래 요청 확인 코드 중 하나다.
        NOT_FOUND: 없는 Lot·설비·공정·작업 지시·Hold
        EQUIPMENT_STEP: 요청 공정에 속하지 않은 설비
        HOLD_CLOSED: 이미 처분된 Hold에 다른 내용의 처분
        RETEST_NOT_AT_INSPECT: 판정을 받는 기간(INSPECT 시작 ~ 다음 공정 투입 전) 밖의 재검사 처분
        LOT_EXISTS: 같은 Lot이 다른 웨이퍼 구성으로 이미 있음
        RESULT_NOT_ACCEPTED: 판정을 받는 기간이 아니거나 그 Lot의 검사 작업 지시가 아님
        ROUND_MISMATCH: 판정 차수가 Lot의 현재 차수와 다름
        INVALID: 그 밖의 잘못된 입력·상태

    Attributes:
        rule: 위반한 규칙 코드.
    """

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"[{rule}] {message}")
        self.rule = rule
