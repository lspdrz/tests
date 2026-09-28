"""Guard: a finished task filed under an empty item id is not kept.

`create_task` files a task under its item id, and `cleanup_task` only removes it for a truthy id,
so a task created with an empty or missing id stayed in the item registry for the life of the
process. Upstream fixed it by filing only under a truthy id (#29980).

Stays a unit test: no route creates a task without an item id today (the chat and subagent paths
always pass a chat id, the note editor a note id), so no request can reach the guarded path. The
rate limiter, rejected avatar and finished reply cases are covered on a running server in
`integration/footprint/test_unbounded_process_state.py`.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3). Unmarked because no issue is filed.
Discriminates: fails in a copy of dev bbfa876af that files tasks under a falsy item id again
(#29980).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


async def _noop(): ...


@pytest.mark.asyncio
@pytest.mark.parametrize("item_id", ["", None])
async def test_a_finished_task_without_an_item_id_is_not_tracked(tasks_module, item_id):
    task_id, task = await tasks_module.create_task(redis=None, coroutine=_noop(), id=item_id)
    await task

    assert task_id not in await tasks_module.list_task_ids_by_item_id(redis=None, id=item_id)
