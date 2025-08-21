"""Test the fixed list_comunas_en_regiones tool handling in ReAct agent.

Usage:
uv run -m src.agents.ReAct_agent.test_list_comunas_fix
"""

import asyncio
import logging

from src.agents.planner_agent.schemas import Plan, Step
from src.agents.ReAct_agent.graph_builder import graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_list_comunas_en_regiones() -> None:
    """Test various instruction formats for list_comunas_en_regiones tool."""
    # Test cases with different instruction formats
    test_cases = [
        {
            "name": "Explicit region marker",
            "instruction": "Listar las comunas de la región: Región de Antofagasta",
        },
        {
            "name": "Natural language with 'en la'",
            "instruction": "Obtener todas las comunas en la Región de Coquimbo",
        },
        {
            "name": "Natural language with 'de la'",
            "instruction": "Mostrar las comunas de la Región Metropolitana de Santiago",
        },
        {
            "name": "Natural language with 'para la'",
            "instruction": "Buscar comunas para la Región de Valparaíso",
        },
        {
            "name": "Mixed case and accents",
            "instruction": "Listar comunas de la región de Tarapacá",
        },
        {
            "name": "Without explicit markers",
            "instruction": "Obtener comunas Región de O'Higgins",
        },
    ]

    logger.info("🧪 Testing list_comunas_en_regiones tool with various instructions")
    logger.info("=" * 60)

    for test_case in test_cases:
        logger.info(f"\n📝 Test: {test_case['name']}")
        logger.info(f"   Instruction: {test_case['instruction']}")

        # Create a simple plan with just one step
        test_plan = Plan(
            goal=f"Test extraction for: {test_case['name']}",
            steps=[
                Step(
                    instruction=test_case["instruction"],
                    suggested_tool="list_comunas_en_regiones",
                    reasoning="Testing region extraction",
                    result="",
                    is_complete=False,
                ),
            ],
            direct_response_to_the_user="",
        )

        try:
            # Execute the plan
            final_state = None
            async for chunk in graph.astream(
                {"plan": test_plan},
                stream_mode="updates",
                debug=False,
            ):
                if "finish" in chunk:
                    final_state = chunk["finish"]

            # Check results
            if final_state and "tool_results" in final_state:
                tool_results = final_state["tool_results"]
                if tool_results:
                    result = tool_results[0].result
                    if "error" in result.lower():
                        logger.error(f"   ❌ Failed: {result[:100]}")
                    else:
                        logger.info(f"   ✅ Success: {result[:100]}...")
                else:
                    logger.warning("   ⚠️  No tool results")
            else:
                logger.error("   ❌ No final state")

        except Exception as e:
            logger.error(f"   ❌ Exception: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Test completed")


async def test_other_tools() -> None:
    """Test that list_comunas and list_regiones work without parameters."""
    logger.info("\n🧪 Testing tools without parameters")
    logger.info("=" * 60)

    # Test list_regiones
    logger.info("\n📝 Testing list_regiones (no parameters)")
    plan_regiones = Plan(
        goal="List all regions",
        steps=[
            Step(
                instruction="Listar todas las regiones disponibles",
                suggested_tool="list_regiones",
                reasoning="Get all regions",
                result="",
                is_complete=False,
            ),
        ],
        direct_response_to_the_user="",
    )

    try:
        final_state = None
        async for chunk in graph.astream(
            {"plan": plan_regiones},
            stream_mode="updates",
            debug=False,
        ):
            if "finish" in chunk:
                final_state = chunk["finish"]

        if final_state and "tool_results" in final_state:
            tool_results = final_state["tool_results"]
            if tool_results:
                result = tool_results[0].result
                logger.info(f"   ✅ Success: {result[:100]}...")
        else:
            logger.error("   ❌ Failed")
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")

    # Test list_comunas
    logger.info("\n📝 Testing list_comunas (no parameters)")
    plan_comunas = Plan(
        goal="List all communes",
        steps=[
            Step(
                instruction="Listar todas las comunas del sistema",
                suggested_tool="list_comunas",
                reasoning="Get all communes",
                result="",
                is_complete=False,
            ),
        ],
        direct_response_to_the_user="",
    )

    try:
        final_state = None
        async for chunk in graph.astream(
            {"plan": plan_comunas},
            stream_mode="updates",
            debug=False,
        ):
            if "finish" in chunk:
                final_state = chunk["finish"]

        if final_state and "tool_results" in final_state:
            tool_results = final_state["tool_results"]
            if tool_results:
                result = tool_results[0].result
                logger.info(f"   ✅ Success: {result[:100]}...")
        else:
            logger.error("   ❌ Failed")
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests completed")


async def main() -> None:
    """Run all tests."""
    await test_list_comunas_en_regiones()
    await test_other_tools()


if __name__ == "__main__":
    asyncio.run(main())
