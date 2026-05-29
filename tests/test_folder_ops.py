from tests.conftest import insert_files, insert_folders


async def test_rename_folder_top_level_collision(client):
    """Renaming to a path that already exists as a folder returns 409."""
    await insert_folders("a", "b")

    res = await client.post("/api/folder-ops/rename", json={"old_path": "a", "new_path": "b"})

    assert res.status_code == 409


async def test_rename_folder_subfolder_collision(client):
    """Renaming 'a' to 'b' returns 409 when 'b/sub' already exists (new preflight check)."""
    # 'b' itself does NOT exist (would trigger the top-level check instead)
    await insert_folders("a", "a/sub", "b/sub")

    res = await client.post("/api/folder-ops/rename", json={"old_path": "a", "new_path": "b"})

    assert res.status_code == 409
    assert "b/sub" in res.json()["detail"]


async def test_rename_folder_success(client):
    """Successful rename updates the folder and all contained file paths."""
    await insert_folders("a", "a/sub")
    await insert_files(("a/file.md", "hello"))

    res = await client.post("/api/folder-ops/rename", json={"old_path": "a", "new_path": "b"})

    assert res.status_code == 200
    updated_filenames = [f["filename"] for f in res.json()]
    assert "b/file.md" in updated_filenames
    assert "a/file.md" not in updated_filenames
