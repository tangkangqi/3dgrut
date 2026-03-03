#!/usr/bin/env python3
"""
Generate USDA file for SimHei text string.

This module provides functions to generate USDA files containing 3D text meshes
from the SimHei font character library.
"""

import re
import os
from typing import Tuple, Dict, Optional


def get_char_unicode_hex(char: str) -> str:
    """Get the Unicode hex code for a character (uppercase, no 0x prefix, padded to 4 digits)."""
    return format(ord(char), '04X')


def sanitize_string(text: str) -> str:
    """Replace non-alphanumeric characters with underscore."""
    return ''.join(c if c.isalnum() else '_' for c in text)


def extract_character_definitions(usda_file_path: str) -> Dict[str, str]:
    """
    Extract character definitions from simhei_grid.usda file.
    
    Returns a dictionary mapping unicode hex codes to their complete Xform definitions.
    """
    char_defs = {}
    
    with open(usda_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Xform definitions for SimHei characters
    # Pattern to find the start: def Xform "SimHei_XXXX"
    xform_pattern = r'def Xform "SimHei_([0-9A-Fa-f]+)"'
    
    matches = list(re.finditer(xform_pattern, content))
    
    for i, match in enumerate(matches):
        unicode_hex = match.group(1).upper()
        start_pos = match.start()
        
        # Find the end of this Xform definition by counting braces
        # Start from the opening brace after the Xform declaration
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            continue
        
        brace_count = 0
        pos = brace_start
        
        while pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the closing brace
                    end_pos = pos + 1
                    full_def = content[start_pos:end_pos]
                    char_defs[unicode_hex] = full_def
                    break
            pos += 1
    
    return char_defs


def get_char_width_from_extent(char_def: str) -> float:
    """
    Extract character width from the extent field in the mesh definition.
    
    Returns the width (x-axis difference) of the character.
    """
    # Find extent line: float3[] extent = [(min_x, min_y, min_z), (max_x, max_y, max_z)]
    extent_pattern = r'float3\[\] extent = \[\(([^,]+),([^,]+),([^)]+)\),\s*\(([^,]+),([^,]+),([^)]+)\)\]'
    match = re.search(extent_pattern, char_def)
    
    if match:
        min_x = float(match.group(1))
        max_x = float(match.group(4))
        return max_x - min_x
    
    # Fallback: if extent not found, use a default width
    # This is approximate based on typical character widths
    return 0.5


def generate_text_usda(
    text: str,
    position: Tuple[float, float, float],
    source_usda_path: str,
    output_usda_path: str,
    char_spacing: float = 0.1
) -> None:
    """
    Generate a USDA file containing 3D text meshes for the given string.
    
    Args:
        text: Input string (non-alphanumeric chars will be replaced with '_')
        position: Starting position (x, y, z) for the text
        source_usda_path: Path to simhei_grid.usda file
        output_usda_path: Path where the output USDA file will be written
        char_spacing: Additional spacing between characters (default: 0.1)
    
    Raises:
        FileNotFoundError: If source_usda_path doesn't exist
        ValueError: If text is empty or contains no valid characters
    """
    if not os.path.exists(source_usda_path):
        raise FileNotFoundError(f"Source USDA file not found: {source_usda_path}")
    
    # Sanitize input string
    sanitized = sanitize_string(text)
    if not sanitized:
        raise ValueError("Text contains no alphanumeric characters")
    
    # Extract character definitions
    print(f"Extracting character definitions from {source_usda_path}...")
    char_defs = extract_character_definitions(source_usda_path)
    print(f"Found {len(char_defs)} character definitions")
    
    # Read the header from source file
    with open(source_usda_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find where the header ends (before first character definition)
    header_end = 0
    for i, line in enumerate(lines):
        if 'def Xform "SimHei_' in line:
            header_end = i
            break
    
    header = ''.join(lines[:header_end])
    
    # Find the text Xform opening (where characters are placed)
    text_xform_start = header.find('def Xform "text"')
    if text_xform_start == -1:
        # Fallback: create a simple header
        header = '''#usda 1.0
(
    customLayerData = {
        dictionary omni_layer = {
            string authoring_layer = "./output.usda"
        }
    }
    defaultPrim = "root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "root"
{
    def Xform "text"
    {
'''
    else:
        # Extract up to and including the text Xform opening
        text_xform_end = header.find('{', text_xform_start) + 1
        header = header[:text_xform_end] + '\n'
    
    # Generate character definitions
    x, y, z = position
    current_x = x
    char_blocks = []
    
    missing_chars = []
    
    for char in sanitized:
        unicode_hex = get_char_unicode_hex(char)
        
        if unicode_hex not in char_defs:
            # Try lowercase
            if unicode_hex.lower() not in char_defs:
                missing_chars.append(f"{char} (U+{unicode_hex})")
                continue
            else:
                unicode_hex = unicode_hex.lower()
        
        char_def = char_defs[unicode_hex]
        
        # Calculate character width
        char_width = get_char_width_from_extent(char_def)
        
        # Update the translate position in the character definition
        # Find the translate line and replace it (handle both float3 and double3)
        translate_pattern = r'((?:float3|double3) xformOp:translate = \()[^)]+\)'
        new_translate = f'\\1({current_x}, {y}, {z})'
        updated_def = re.sub(translate_pattern, new_translate, char_def)
        
        # Ensure proper indentation (8 spaces for characters inside text Xform)
        lines = updated_def.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():  # Skip empty lines
                # Remove existing leading whitespace and add 8 spaces
                indented_lines.append('        ' + line.lstrip())
            else:
                indented_lines.append('')
        
        char_blocks.append('\n'.join(indented_lines))
        
        # Move to next character position
        current_x += char_width + char_spacing
    
    if missing_chars:
        print(f"Warning: Missing characters: {', '.join(missing_chars)}")
    
    if not char_blocks:
        raise ValueError("No valid characters found in text")
    
    # Find material definition in source file
    material_def = ""
    material_pattern = r'(def Scope "_materials"[^}]*def Material[^}]*\{[^}]*\}[^}]*\})'
    with open(source_usda_path, 'r', encoding='utf-8') as f:
        content = f.read()
        match = re.search(material_pattern, content, re.DOTALL)
        if match:
            material_def = match.group(1)
    
    # Write output file
    with open(output_usda_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write('\n'.join(char_blocks))
        f.write('\n')
        if material_def:
            f.write('        ' + material_def.replace('\n', '\n        '))
            f.write('\n')
        f.write('    }\n')  # Close text Xform
        f.write('}\n')  # Close root Xform
    
    print(f"Generated USDA file: {output_usda_path}")
    print(f"Text: '{sanitized}'")
    print(f"Position: ({x}, {y}, {z})")
    print(f"Characters: {len(char_blocks)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python generate_text_usda.py <text> <x> <y> <z> [output_path]")
        print("Example: python generate_text_usda.py 'Hello123' 0 0 0 output.usda")
        sys.exit(1)
    
    text = sys.argv[1]
    x, y, z = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    output_path = sys.argv[5] if len(sys.argv) > 5 else "text_output.usda"
    
    source_path = os.path.join(
        os.path.dirname(__file__),
        "interio",
        "simhei_grid.usda"
    )
    
    try:
        generate_text_usda(
            text=text,
            position=(x, y, z),
            source_usda_path=source_path,
            output_usda_path=output_path
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

