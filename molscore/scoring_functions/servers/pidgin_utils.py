import os
from pathlib import Path
from typing import Callable, Sequence, Union

import pystow
from zenodo_client import Zenodo as ZenodoBase

os.environ["PYSTOW_NAME"] = ".pidgin_data"


class Zenodo(ZenodoBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_latest_record(self, record_id: Union[int, str]) -> str:
        """Get the latest record related to the given record."""
        # res_json = self.get_record(record_id).json()
        # Still works even in the case that the given record ID is the latest.
        # latest = res_json["links"]["latest"].split("/")[-1]
        latest = os.path.join(
            record_id, "versions", "latest"
        )  ##### Change in Zenodo API
        return latest

    def download(
        self,
        record_id: Union[int, str],
        name: str,
        *,
        force: bool = False,
        parts: Union[
            None, Sequence[str], Callable[[str, str, str], Sequence[str]]
        ] = None,
    ) -> Path:
        """Download the file for the given record.

        :param record_id: The Zenodo record id
        :param name: The name of the file in the Zenodo record
        :param parts: Optional arguments on where to store with :func:`pystow.ensure`. If none given, goes in
            ``<PYSTOW_HOME>/zendoo/<CONCEPT_RECORD_ID>/<RECORD>/<PATH>``. Where ``CONCEPT_RECORD_ID`` is the
            consistent concept record ID for all versions of the same record. If a function is given, the function
            should take 3 position arguments: concept record id, record id, and version, then return a sequence for
            PyStow. The name of the file is automatically appended to the end of the sequence.
        :param force: Should the file be re-downloaded if it already is cached? Defaults to false.
        :returns: the path to the downloaded file.
        :raises FileNotFoundError: If the Zenodo record doesn't have a file with the given name

        For example, to download the most recent version of NSoC-KG, you can
        use the following command:

        >>> path = Zenodo().download('4574555', 'triples.tsv')

        Even as new versions of the data are uploaded, this command will always
        be able to check if a new version is available, download it if it is, and
        return the local file path. If the most recent version is already downloaded,
        then it returns the local file path to the cached file.

        The file path uses :mod:`pystow` under the ``zenodo`` module and uses the
        "concept record ID" as a submodule since that is the consistent identifier
        between different records that are versions of the same data.
        """
        res_json = self.get_record(record_id).json()
        # conceptrecid is the consistent record ID for all versions of the same record
        concept_record_id = res_json["conceptrecid"]
        # FIXME send error report to zenodo about this - shouldn't version be required?
        version = res_json["metadata"].get("version", "v1")

        try:
            for file in res_json["files"]:
                if (
                    file["filename"] == name
                ):  ##### Change in Zenodo API, "filename" not "key"
                    url = os.path.join(
                        file["links"]["self"].rsplit("/", 1)[0], name, "content"
                    )  ##### file["links"]["self"] no longer works instead try e.g., https://zenodo.org/records/7547691/files/trained_models.zip/content
                    break
            else:
                raise FileNotFoundError(
                    f"zenodo.record:{record_id} does not have a file with key {name}"
                )
        except Exception:
            for file in res_json["files"]:
                if file["key"] == name:
                    url = file["links"]["self"]
                    break
            else:
                raise FileNotFoundError(
                    f"zenodo.record:{record_id} does not have a file with key {name}"
                )

        if parts is None:
            parts = [self.module.replace(":", "-"), concept_record_id, version]
        elif callable(parts):
            parts = parts(concept_record_id, str(record_id), version)
        return pystow.ensure(*parts, name=name, url=url, force=force)
