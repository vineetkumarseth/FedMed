from fedmed.audit import static_server_check


def test_server_never_imports_the_data_package():
    r = static_server_check()
    assert r["passed"], r["detail"]
