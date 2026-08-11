from kawaneen.corpus.ids import canonical_id


def test_uuidv5_ids_are_stable_and_depend_on_location_not_text() -> None:
    first = canonical_id("arabiccr", "3", "ArabiCCR-dataset.csv", 7, "case_text")
    second = canonical_id("arabiccr", "3", "ArabiCCR-dataset.csv", 7, "case_text")
    other_field = canonical_id("arabiccr", "3", "ArabiCCR-dataset.csv", 7, "RULING")
    assert first == second
    assert first != other_field
    assert len(first) == 36
