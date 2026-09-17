import pytest

from agents.supervisor import SupervisorAgent
from config.settings import Settings


@pytest.mark.asyncio
async def test_mini_omits_temperature(mock_openai):
    mock_openai['set_response']({})
    agent = SupervisorAgent()
    agent._model = 'gpt-5-mini'
    await agent.run('Fictional research case')
    arguments = mock_openai['create_mock'].call_args.kwargs
    assert 'temperature' not in arguments
    assert arguments['reasoning_effort'] == 'minimal'


def test_settings_accept_unrelated_environment(tmp_path):
    env = tmp_path / '.env'
    env.write_text('UNRELATED_EVALUATION_SETTING=test\n', encoding='utf-8')
    assert Settings(_env_file=env).OPENAI_MODEL
