from __future__ import annotations
from pathlib import Path


class PvdCollectionWriter:
    """
    Manages writing ParaView Data (.pvd) collection files linking time-series VTU files.
    """

    def __init__(self, pvd_path: Path | str) -> None:
        self.pvd_path = Path(pvd_path)
        self.entries: list[tuple[float, str]] = []

    def add_timestep(self, timestep: float, filename: str) -> None:
        """Appends a timestep entry to the collection."""
        self.entries.append((timestep, filename))

    def write(self) -> None:
        """Writes the XML collection to the target .pvd file."""
        self.pvd_path.parent.mkdir(parents=True, exist_ok=True)
        xml_content = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>',
        ]
        for timestep, filename in self.entries:
            xml_content.append(
                f'    <DataSet timestep="{timestep:.6e}" group="" part="0" file="{filename}"/>'
            )
        xml_content.append('  </Collection>')
        xml_content.append('</VTKFile>')

        self.pvd_path.write_text('\n'.join(xml_content))
