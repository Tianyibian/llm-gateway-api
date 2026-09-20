import csv

import pytest

from app.services.graph_data import export_catalog, prepare_catalog


@pytest.fixture
def catalog(tmp_path):
    fixtures = {
        "Products": [["ProductID", "ProductName", "SupplierID", "CategoryID", "UnitPrice"],
                     ["1", "Acme Sensor", "7", "2", "9999"]],
        "Suppliers": [["SupplierID", "CompanyName", "ContactName", "Phone"],
                      ["7", "Acme Supply", "Private Person", "secret-phone"]],
        "Categories": [["CategoryID", "CategoryName", "Description"],
                       ["2", "Sensor", "unreviewed nonsense"]],
    }
    for name, rows in fixtures.items():
        with (tmp_path / f"{name}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    return tmp_path


def test_catalog_joins_only_real_foreign_keys_and_keeps_provenance(catalog):
    data = prepare_catalog(catalog)
    assert len(data.nodes) == 3 and len(data.relationships) == 2
    assert data.relationships[0] == {
        "source": "Product:1", "type": "SUPPLIED_BY", "target": "Supplier:7",
        "source_column": "Products.SupplierID",
    }
    assert all(len(node["source_sha256"]) == 64 for node in data.nodes)
    for value in ["Private Person", "secret-phone", "unreviewed nonsense", "9999"]:
        assert value not in str(data)
    assert "Acme Supply" in data.documents["product-1.txt"]


def test_export_creates_review_artifacts_not_a_fake_index(catalog):
    data = prepare_catalog(catalog)
    target = catalog / "seed"
    report = export_catalog(data, target)
    assert report["index_built"] is False
    assert report["review_required_before_cloud_indexing"] is True
    assert (target / "input/product-1.txt").is_file()
    with pytest.raises(FileExistsError):
        export_catalog(data, target)


@pytest.mark.parametrize("replacement,match", [
    ("1,Acme Sensor,999,2,10\n", "foreign key"),
    ("1,Acme Sensor,7,2,10\n1,Other Sensor,7,2,10\n", "Duplicate"),
    ("../escape,Acme Sensor,7,2,10\n", "identifier"),
    ("1,,7,2,10\n", "Blank"),
])
def test_bad_source_data_fails_before_export(catalog, replacement, match):
    (catalog / "Products.csv").write_text("ProductID,ProductName,SupplierID,CategoryID,UnitPrice\n" + replacement)
    with pytest.raises(ValueError, match=match):
        prepare_catalog(catalog)


def test_preparation_is_deterministic(catalog):
    assert prepare_catalog(catalog) == prepare_catalog(catalog)
