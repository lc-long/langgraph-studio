import pytest


class TestSupervisorService:
    def test_supervisor_state_uses_annotated_reducers(self):
        from app.services.supervisor_service import SupervisorState, last_write_wins
        import operator
        hints = SupervisorState.__annotations__
        # messages uses operator.add (list concatenation)
        assert hints["messages"] == Annotated[list, operator.add]
        # nextAgent and step use last_write_wins
        assert hints["nextAgent"] == Annotated[str, last_write_wins]
        assert hints["step"] == Annotated[str, last_write_wins]
        # completedAgents uses operator.add (list concatenation)
        assert hints["completedAgents"] == Annotated[list[str], operator.add]

    def test_last_write_wins_reducer(self):
        from app.services.supervisor_service import last_write_wins
        assert last_write_wins("a", "b") == "b"
        assert last_write_wins("older", "newer") == "newer"


class TestCodeReviewService:
    def test_code_review_state_structure(self):
        from app.services.code_review_service import ReviewState
        assert "code" in ReviewState.__annotations__
        assert "reviewResults" in ReviewState.__annotations__
        assert "report" in ReviewState.__annotations__


class TestParallelService:
    def test_parallel_state_has_execution_log(self):
        from app.services.parallel_service import ParallelState
        assert "task" in ParallelState.__annotations__
        assert "results" in ParallelState.__annotations__
        assert "finalReport" in ParallelState.__annotations__
        assert "executionLog" in ParallelState.__annotations__
        # Verify Annotated with operator.add
        import operator
        from typing import Annotated
        assert ParallelState.__annotations__["executionLog"] == Annotated[list[str], operator.add]


class TestTechResearchService:
    def test_tech_research_state_structure(self):
        from app.services.tech_research_service import TechResearchState
        assert "question" in TechResearchState.__annotations__
        assert "researchResults" in TechResearchState.__annotations__
        assert "report" in TechResearchState.__annotations__


class TestLLMFactory:
    def test_llm_factory_returns_client(self):
        from app.services.llm_factory import get_llm
        llm = get_llm()
        assert llm is not None
        assert hasattr(llm, "invoke")


class TestReactAgentService:
    def test_tools_registered(self):
        from app.services.react_agent_service import tools
        tool_names = [t.name for t in tools]
        assert "calculator" in tool_names
        assert "get_weather" in tool_names
        # 原有2个工具 + 新增4个工具 = 6个
        assert "web_search" in tool_names
        assert "web_content_fetch" in tool_names
        assert "file_read" in tool_names
        assert "code_interpreter" in tool_names
        assert len(tools) == 6