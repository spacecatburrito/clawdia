from clawdia.kinds import WORK_KINDS, kind_from_hashtags


def test_hashtags_english_russian_case_insensitive():
    assert kind_from_hashtags("Check this #Job") == "job"
    assert kind_from_hashtags("#рецепт борща") == "recipe"
    assert kind_from_hashtags("no tags here") is None
    assert kind_from_hashtags("#idea #task") == "idea"  # first match wins


def test_work_kinds():
    assert WORK_KINDS == {"task", "job", "event"}


def test_all_aliases_map():
    table = {
        "task": ["#task", "#work", "#задача"],
        "job": ["#job", "#vacancy", "#вакансия"],
        "event": ["#event", "#ивент"],
        "recipe": ["#recipe", "#рецепт"],
        "film": ["#film", "#movie", "#фильм"],
        "place": ["#place", "#место"],
        "buy": ["#buy", "#shop", "#купить"],
        "idea": ["#idea", "#идея"],
    }
    for kind, tags in table.items():
        for tag in tags:
            assert kind_from_hashtags(f"x {tag.upper()} y") == kind


def test_hashtag_must_be_whole_word():
    assert kind_from_hashtags("#tasks and #jobless") is None
    assert kind_from_hashtags("see #task.") == "task"
    assert kind_from_hashtags("c#task") is None
