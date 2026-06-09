"""Phần 5 homework: trace_id flow (5.1) và ghi chú test Tax Agent down (5.2).

Chạy sau khi đã chạy test_client.py và có trace_id trong service logs.
"""

STAGE5_SEQUENCE = """
sequenceDiagram
    participant User as test_client
    participant Customer as CustomerAgent_10100
    participant Registry as Registry_10000
    participant Law as LawAgent_10101
    participant Tax as TaxAgent_10102
    participant Compliance as ComplianceAgent_10103

    User->>Customer: A2A message (question)
    Note over Customer: trace_id generated here
    Customer->>Customer: LLM delegate_to_legal_agent
    Customer->>Registry: discover legal_question
    Registry-->>Customer: Law Agent URL
    Customer->>Law: A2A delegate (trace_id, depth=1)
    Law->>Law: analyze_law (LLM)
    Law->>Law: route_to_specialists (keyword)
    par parallel
        Law->>Registry: discover tax_question
        Registry-->>Law: Tax Agent URL
        Law->>Tax: A2A delegate (trace_id, depth=2)
        Tax->>Tax: ReAct agent (LLM)
        Tax-->>Law: tax analysis
    and
        Law->>Registry: discover compliance_question
        Registry-->>Law: Compliance Agent URL
        Law->>Compliance: A2A delegate (trace_id, depth=2)
        Compliance->>Compliance: ReAct agent (LLM)
        Compliance-->>Law: compliance analysis
    end
    Law->>Law: aggregate (LLM)
    Law-->>Customer: final_answer
    Customer-->>User: RESPONSE + Latency
"""


def print_homework_5_1() -> None:
    print("=" * 70)
    print("BAI TAP 5.1: Request flow voi trace_id")
    print("=" * 70)
    print("""
Cach trace:
  1. Chay: uv run python test_client.py
  2. Trong log Customer/Law/Tax/Compliance, tim cung trace_id (UUID)
     Vi du log: "trace=d5b9d581-3d3d-4fe2-b760-4e6fac59d627"
  3. trace_id sinh tai Customer Agent, propagate qua moi hop A2A delegate

Sequence diagram (mermaid):
""")
    print(STAGE5_SEQUENCE)


def print_homework_5_2() -> None:
    print("=" * 70)
    print("BAI TAP 5.2: Test dynamic discovery (Tax Agent down)")
    print("=" * 70)
    print("""
Cach test:
  1. Dung Tax Agent (Ctrl+C cua cua so tax_agent hoac kill process port 10102)
  2. Chay lai: uv run python test_client.py
  3. Quan sat: Law Agent van chay, call_tax tra ve loi/graceful message
     "[Tax analysis unavailable: ...]" trong tax_result
  4. He thong van tra loi (compliance + law analysis) nhung thieu phan tax

Ket qua mong doi: khong crash toan he thong; tax_result chua thong bao loi.

Ket qua thuc te (da chay):
  - Tax Agent down -> test_client van tra loi trong ~10.57s
  - He thong khong crash; aggregate van tong hop law + compliance
""")


if __name__ == "__main__":
    print_homework_5_1()
    print_homework_5_2()
