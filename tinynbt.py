
""" The library is only support Minecraft 1.21.x for write map nbt file"""

import struct
import gzip

# --- NBT Tag Types ---
TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class NBTTag:
    """NBT Basic Class"""

    def __init__(self, tag_type, name=""):
        self.tag_type = tag_type
        self.name = name.encode('utf-8') if isinstance(name, str) else name

    def _write_name(self, stream):
        if self.name:
            stream.write(struct.pack('>H', len(self.name)))
            stream.write(self.name)
        else:
            stream.write(struct.pack('>H', 0))

    def write_payload(self, stream):
        raise NotImplementedError("Subclasses must implement write_payload")

    def write(self, stream):
        stream.write(struct.pack('B', self.tag_type))
        if self.tag_type != TAG_END:
            self._write_name(stream)
            self.write_payload(stream)


class TAG_Byte(NBTTag):
    def __init__(self, name, value):
        super().__init__(TAG_BYTE, name)
        # Clamp the value to signed byte range for TAG_Byte
        self.value = max(-128, min(127, int(value)))

    def write_payload(self, stream):
        stream.write(struct.pack('b', self.value))


class TAG_Int(NBTTag):
    def __init__(self, name, value):
        super().__init__(TAG_INT, name)
        self.value = value

    def write_payload(self, stream):
        stream.write(struct.pack('>i', self.value))


class TAG_Byte_Array(NBTTag):
    """Represents an NBT Byte Array Tag."""

    def __init__(self, name, value):
        """
        Args:
            name (str): The name of the tag.
            value (list of int, bytes, or bytearray):
                The byte data.
                - If list/tuple: values should be integers. They will be interpreted as unsigned
                  bytes (0-255) and stored as the corresponding signed byte (-128 to 127)
                  internally by bytearray. The `write_payload` method handles this correctly
                  for the NBT format.
        """
        super().__init__(TAG_BYTE_ARRAY, name)

        # --- FIXED: Correctly handle list/tuple of unsigned byte values (0-255) ---
        if isinstance(value, (list, tuple)):
            # Ensure all values are integers within 0-255 range before creating bytearray.
            # bytearray() constructor requires 0 <= v <= 255. It handles the
            # signed/unsigned conversion internally: 128-255 become -128 to -1.
            try:
                # Use a generator for memory efficiency
                self.value = bytearray(v & 0xFF for v in value)
            except ValueError as e:
                # This should ideally not happen with `v & 0xFF`, but re-raise for clarity
                raise ValueError(f"Invalid byte value provided to TAG_Byte_Array '{name}': {e}") from e
        # --- ---
        elif isinstance(value, (bytes, bytearray)):
            # If already bytes/bytearray, store a copy
            self.value = bytearray(value)
        else:
            raise TypeError("TAG_Byte_Array value must be a list, tuple, bytes, or bytearray")

    def write_payload(self, stream):
        """Writes the payload of the tag to the stream."""
        # The length of the byte array is written as a 4-byte integer
        stream.write(struct.pack('>i', len(self.value)))
        # The raw bytes are written directly. Python's bytearray correctly
        # stores 128-255 as -128 to -1 (signed), which is the correct NBT byte representation
        # for unsigned values 128-255.
        stream.write(self.value)


class TAG_Compound(NBTTag):
    def __init__(self, name, value):
        """
        Args:
            name (str): The name of the compound tag.
            value (dict or iterable of NBTTag):
                A dictionary mapping tag names (str) to NBTTag objects,
                or an iterable of NBTTag objects.
        """
        super().__init__(TAG_COMPOUND, name)
        if isinstance(value, dict):
            self.value = value
        else:
            # If it's not a dict, assume it's an iterable of NBTTag objects
            # and build a dict from their names.
            self.value = {tag.name.decode('utf-8') if isinstance(tag.name, bytes) else tag.name: tag for tag in value}

    def write_payload(self, stream):
        # Iterate over the NBTTag objects in the dictionary's values
        for tag in self.value.values():
            # Write each tag using its own `write` method
            tag.write(stream)
        # Write TAG_End (0x00) to signify the end of the compound tag
        stream.write(struct.pack('B', TAG_END))


def write_nbt_file(root_tag, filename, gzipped=True):
    """
    Write an NBT tag to a file.
    Args:
        root_tag (TAG_Compound): The root compound tag to write.
        filename (str): The path to the output .dat file.
        gzipped (bool): If True, the file will be gzip compressed.
    """
    if not isinstance(root_tag, TAG_Compound):
        raise TypeError("Root tag must be a TAG_Compound")

    mode = 'wb'  # Mode is 'wb' for both, gzip.open handles compression
    opener = gzip.open if gzipped else open

    with opener(filename, mode) as f:
        # For NBT files, the root tag type and name are written first
        # Root tag type
        f.write(struct.pack('B', root_tag.tag_type))
        # Root tag name (can be empty for root)
        root_tag._write_name(f)
        # Payload of root tag (its children)
        root_tag.write_payload(f)


# --- Convenience Functions for Map Data ---
def create_map_nbt(color_data, map_scale=2, map_dimension=0, x_center=0, z_center=0):
    """
    Build Map NBT Data (Minimal).
    Args:
        color_data (list of int): 16384 (128x128) integers representing map colors (0-255).
        map_scale (int): The scale of the map (default 2).
        map_dimension (int): The dimension (0=Overworld, etc.).
        x_center (int): The center X coordinate.
        z_center (int): The center Z coordinate.
    Returns:
        TAG_Compound: The root NBT tag for the map.
    """
    if len(color_data) != 128 * 128:
        raise ValueError("color_data must contain exactly 16384 elements.")

    # Create the root compound tag (maps don't have a "data" wrapper in the file)
    # The tags are direct children of the root.
    root_compound = TAG_Compound("", {  # Root name is typically empty for map .dat files
        "DataVersion": TAG_Int("DataVersion", 3953),  # DataVersion for 1.21.x
        "data": TAG_Compound("data", {
            "version": TAG_Int("version", 19133),  # Map data version
            "dimension": TAG_Byte("dimension", map_dimension),
            "xCenter": TAG_Int("xCenter", x_center),
            "zCenter": TAG_Int("zCenter", z_center),
            "scale": TAG_Byte("scale", map_scale),
            "locked": TAG_Byte("locked", 0),  # Explicitly set locked to 0
            # The colors tag - this is where the fix is most important
            "colors": TAG_Byte_Array("colors", color_data),
            "trackingPosition": TAG_Byte("trackingPosition",0)
        })
    })

    return root_compound  # This is the root tag
