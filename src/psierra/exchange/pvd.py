from __future__ import annotations
from pathlib import Path


class PvdCollectionWriter:
    """
    Manages writing ParaView Data (.pvd) collection files linking time-series or multi-domain VTU files.
    """

    def __init__(self, pvd_path: Path | str) -> None:
        self.pvd_path = Path(pvd_path)
        self.entries: list[dict[str, str | float | int | None]] = []

    def add_timestep(
        self,
        timestep: float,
        filename: str,
        part: int = 0,
        group: str = "",
    ) -> None:
        """Appends a timestep entry to the collection."""
        self.entries.append(
            {
                "timestep": timestep,
                "filename": filename,
                "part": part,
                "group": group,
            }
        )

    def add_dataset(
        self,
        filename: str,
        timestep: float | None = None,
        part: int = 0,
        group: str = "",
    ) -> None:
        """Appends a dataset (e.g. multi-domain component) to the collection."""
        self.entries.append(
            {
                "timestep": timestep,
                "filename": filename,
                "part": part,
                "group": group,
            }
        )

    def write(self) -> None:
        """Writes the XML collection to the target .pvd file."""
        self.pvd_path.parent.mkdir(parents=True, exist_ok=True)
        xml_content = [
            "<?xml version=\"1.0\"?>",
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">",
            "  <Collection>",
        ]
        for entry in self.entries:
            ts = entry["timestep"]
            ts_str = f" timestep=\"{ts:.6e}\"" if ts is not None else ""
            group = entry["group"]
            group_str = f" group=\"{group}\"" if group else ""
            part_str = f" part=\"{entry['part']}\""
            file_str = f" file=\"{entry['filename']}\""
            xml_content.append(f"    <DataSet{ts_str}{group_str}{part_str}{file_str}/>")

        xml_content.append("  </Collection>")
        xml_content.append("</VTKFile>")

        self.pvd_path.write_text("\n".join(xml_content))
