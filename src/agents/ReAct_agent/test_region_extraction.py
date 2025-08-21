"""Simple test for the region extraction logic without dependencies.

Usage:
python src/agents/ReAct_agent/test_region_extraction.py
"""

import re


def extract_region(instruction: str) -> str | None:
    """Extract region from instruction using the same logic as in agent_logic.py."""
    region_value = None

    # Pattern 1: Look for explicit markers
    marker_variants = [
        "region:",
        "región:",
        "Region:",
        "Región:",
        "para la ",
        "de la ",
        "en la ",
        "para ",
        "de ",
        "en ",
    ]

    for mk in marker_variants:
        if mk.lower() in instruction.lower():
            # Extract text after the marker
            idx = instruction.lower().index(mk.lower())
            potential_region = instruction[idx + len(mk) :].strip()

            # Clean up the extracted text
            # Remove quotes, periods, commas
            potential_region = potential_region.strip("\"'.,")

            # Take the first segment before any punctuation
            for delimiter in [",", ".", "?", "!", "\n"]:
                if delimiter in potential_region:
                    potential_region = potential_region.split(delimiter)[0]

            # Check if we got a reasonable region name
            if potential_region and len(potential_region) > 3:
                region_value = potential_region.strip()
                break

    # Pattern 2: Look for known region patterns
    if not region_value:
        # Common Chilean region patterns
        region_patterns = [
            r"Región\s+de\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)",
            r"Región\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)",
            r"región\s+de\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)",
            r"región\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)",
        ]
        for pattern in region_patterns:
            match = re.search(pattern, instruction)
            if match:
                region_value = match.group(1).strip()
                break

    return region_value


def test_extraction():
    """Test various instruction formats."""
    test_cases = [
        (
            "Listar las comunas de la región: Región de Antofagasta",
            "Región de Antofagasta",
        ),
        ("Obtener todas las comunas en la Región de Coquimbo", "Región de Coquimbo"),
        (
            "Mostrar las comunas de la Región Metropolitana de Santiago",
            "Región Metropolitana de Santiago",
        ),
        ("Buscar comunas para la Región de Valparaíso", "Región de Valparaíso"),
        ("Listar comunas de la región de Tarapacá", "Tarapacá"),
        ("Obtener comunas Región de O'Higgins", "O'Higgins"),
        (
            "Si la región existe, obtener los nombres de todas las comunas de la región.",
            None,
        ),  # Should fail
        ("Lista comunas en Región del Biobío", "Región del Biobío"),
    ]

    print("🧪 Testing region extraction logic")
    print("=" * 60)

    passed = 0
    failed = 0

    for instruction, expected in test_cases:
        result = extract_region(instruction)

        # Check if extraction matches expected
        if expected is None:
            success = result is None
        else:
            success = result == expected or (result and expected in result)

        status = "✅" if success else "❌"

        print(f"\n{status} Instruction: {instruction[:60]}...")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")

        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    test_extraction()
