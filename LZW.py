#!/usr/bin/env python3
"""
MEF University COMP 204 Programming Studio Project, Prof. Muhittin GÖKMEN
Image and Text Compression & Decompression with LZW Algorithm by Mustafa Garip (Using Bit-Level Packaging, 16-bit default)

-------------------------------------------------------------------------------

This program compresses an image (.png or .bmp) or a text file (.txt) using the LZW algorithm,
saves the compressed file, and decompresses it to restore the original content.
Bit-level packing is applied for both text and images.
The default code_length=16 minimizes dictionary overflow issues.

Libraries:
  - Pillow (PIL)
  - tkinter
  - pickle
  - math
"""

import os
import pickle
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

#Bit-Level Packaging
def int_array_to_binary_string(int_array, code_length):
    """Converts the integer list into a bit string with the specified code_length"""

    return "".join(format(num, f'0{code_length}b') for num in int_array)

def pad_encoded_text(encoded_text):
    """Pads the bit string so its length is a multiple of 8. Adds an 8-bit padding size info at the beginning."""

    extra_padding = (8 - len(encoded_text) % 8) % 8
    padded_info = format(extra_padding, '08b')
    padded_text = padded_info + encoded_text + ("0" * extra_padding)
    return padded_text

def get_byte_array(padded_encoded_text):
    """Converts the padded bit string into a bytearray."""

    return bytearray(int(padded_encoded_text[i:i+8], 2) for i in range(0, len(padded_encoded_text), 8))

def remove_padding(padded_encoded_text):
    """Reads the first 8-bit padding info and removes the padding to return the original bit string."""

    extra_padding = int(padded_encoded_text[:8], 2)
    return padded_encoded_text[8:-extra_padding] if extra_padding != 0 else padded_encoded_text[8:]

def binary_string_to_int_array(bitstr, code_length):
    """Splits the bit string into fixed-size pieces of code_length and converts them into an integer list."""

    return [int(bitstr[i:i+code_length], 2) for i in range(0, len(bitstr), code_length)]

#Compression Metrics
def calculate_compression_metrics(original_path, compressed_path):
    """Calculates the original and compressed file size along with the compression ratio."""

    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    ratio = compressed_size / original_size if original_size else 0
    
    return original_size, compressed_size, ratio

def calculate_entropy(data):
    """Calculates the entropy (bits per symbol) for the given data."""

    freq = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(data)
    
    return -sum((count/total) * math.log2(count/total) for count in freq.values())

def show_compression_metrics(original_path, compressed_path):
    """Returns a text containing metrics related to file sizes and entropy."""

    orig_size, comp_size, ratio = calculate_compression_metrics(original_path, compressed_path)

    with open(original_path, "rb") as f:
        data = f.read()

    try:
        data = data.decode("utf-8")
    except UnicodeDecodeError:
        data = data.hex()

    entropy_val = calculate_entropy(data)

    return (f"Original Size: {orig_size} bytes\n"
            f"Compressed Size: {comp_size} bytes\n"
            f"Compression Ratio: {ratio:.3f}\n"
            f"Entropy: {entropy_val:.2f} bits/symbol")

#LZW Compression / Decompression
def lzw_compress(uncompressed):
    """Compresses the given string using the LZW algorithm and returns an integer code list."""

    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []

    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result

def lzw_decompress(compressed):
    """Reconstructs the original string from the LZW code list."""

    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    w = chr(compressed.pop(0))
    result = [w]

    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError(f"Invalid compressed code: {k}")

        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry
    return "".join(result)

#Text File Compression & Decompression (Bit-Level Packaging)
def compress_text_file(filepath, output_path, code_length=16):
    """Compresses a text file using the LZW algorithm and bit-level packing."""

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    codes = lzw_compress(text)
    bit_string = int_array_to_binary_string(codes, code_length)
    padded = pad_encoded_text(bit_string)
    byte_array = get_byte_array(padded)

    data = {
        "type": "text",
        "code_length": code_length,
        "num_codes": len(codes),
        "data": byte_array
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    return output_path

def decompress_text_file(filepath, output_txt_path, code_length=16):
    """Decompresses the compressed text file and writes the original text to a .txt file."""

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "text":
        raise ValueError("This file does not contain text compression!")
    
    code_length = data.get("code_length", code_length)
    byte_array = data["data"]
    bit_str = "".join(format(byte, '08b') for byte in byte_array)
    unpadded = remove_padding(bit_str)
    codes = binary_string_to_int_array(unpadded, code_length)
    text = lzw_decompress(codes)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_txt_path

#Image Pixel Processing
def get_image_pixels(img, use_diff=False):
    """Extracts pixel data from the given PIL image object. If use_diff=True, the Difference method is applied."""

    width, height = img.size
    pixels = list(img.getdata())

    if not use_diff:
        return pixels, width, height
    
    diff_pixels = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0:
                diff_pixels.append(pixels[idx])
            elif x == 0:
                diff = (pixels[idx] - pixels[idx - width]) % 256
                diff_pixels.append(diff)
            else:
                diff = (pixels[idx] - pixels[idx - 1]) % 256
                diff_pixels.append(diff)

    return diff_pixels, width, height

def reconstruct_pixels(diff_pixels, width, height, use_diff=False):
    """Recomputes the original pixel values from the compressed data using the Difference method."""

    if not use_diff:
        return diff_pixels
    
    pixels = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0:
                pixels.append(diff_pixels[idx])
            elif x == 0:
                original = (pixels[idx - width] + diff_pixels[idx]) % 256
                pixels.append(original)
            else:
                original = (pixels[idx - 1] + diff_pixels[idx]) % 256
                pixels.append(original)
    return pixels

#Grayscale Image Compression & Decompression (Bit-Level Packaging)
def compress_gray_image(filepath, output_path, use_diff=False, code_length=16):
    """Opens and converts the input image to grayscale mode, then compresses the pixel values using the LZW algorithm and bit-level packing."""

    img = Image.open(filepath).convert("L")
    pixels, width, height = get_image_pixels(img, use_diff)
    pixel_string = "".join(chr(p) for p in pixels)
    codes = lzw_compress(pixel_string)
    bit_string = int_array_to_binary_string(codes, code_length)
    padded = pad_encoded_text(bit_string)
    byte_array = get_byte_array(padded)

    data = {
        "type": "gray",
        "size": img.size,
        "use_diff": use_diff,
        "code_length": code_length,
        "num_codes": len(codes),
        "data": byte_array
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return output_path

def decompress_gray_image(filepath, output_image_path, code_length=16):
    """Decompresses the compressed grayscale image and saves it in PNG format."""

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "gray":
        raise ValueError("This file does not contain grayscale image compression!")
    
    code_length = data.get("code_length", code_length)
    size = data["size"]
    use_diff = data.get("use_diff", False)
    byte_array = data["data"]
    bit_str = "".join(format(byte, '08b') for byte in byte_array)
    unpadded = remove_padding(bit_str)
    codes = binary_string_to_int_array(unpadded, code_length)
    decompressed_string = lzw_decompress(codes)
    pixels = [ord(c) for c in decompressed_string]

    if use_diff:
        pixels = reconstruct_pixels(pixels, size[0], size[1], use_diff)

    img = Image.new("L", size)
    img.putdata(pixels)
    img.save(output_image_path)

    return output_image_path

#Color Image Compression & Decompression (Bit-Level Packaging)
def compress_color_image(filepath, output_path, use_diff=False, code_length=16):
    """Compresses the color image (RGB) separately for each channel using the LZW algorithm and bit-level packing. The Difference method can be applied optionally."""

    img = Image.open(filepath).convert("RGB")
    width, height = img.size
    r_img, g_img, b_img = img.split()

    if use_diff:
        pixels_r, _, _ = get_image_pixels(r_img, use_diff)
        pixels_g, _, _ = get_image_pixels(g_img, use_diff)
        pixels_b, _, _ = get_image_pixels(b_img, use_diff)
    else:
        pixels_r = list(r_img.getdata())
        pixels_g = list(g_img.getdata())
        pixels_b = list(b_img.getdata())

    str_r = "".join(chr(p) for p in pixels_r)
    str_g = "".join(chr(p) for p in pixels_g)
    str_b = "".join(chr(p) for p in pixels_b)
    codes_r = lzw_compress(str_r)
    codes_g = lzw_compress(str_g)
    codes_b = lzw_compress(str_b)

    bin_r = int_array_to_binary_string(codes_r, code_length)
    bin_g = int_array_to_binary_string(codes_g, code_length)
    bin_b = int_array_to_binary_string(codes_b, code_length)

    padded_r = pad_encoded_text(bin_r)
    padded_g = pad_encoded_text(bin_g)
    padded_b = pad_encoded_text(bin_b)

    ba_r = get_byte_array(padded_r)
    ba_g = get_byte_array(padded_g)
    ba_b = get_byte_array(padded_b)

    data = {
        "type": "color",
        "size": (width, height),
        "use_diff": use_diff,
        "code_length": code_length,
        "num_codes_r": len(codes_r),
        "num_codes_g": len(codes_g),
        "num_codes_b": len(codes_b),
        "data_r": ba_r,
        "data_g": ba_g,
        "data_b": ba_b
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return output_path

def decompress_color_image(filepath, output_image_path, code_length=16):
    """Decompresses the compressed color image; decodes each channel and reconstructs the original RGB image."""

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "color":
        raise ValueError("This file does not contain colored image compression!")
    
    code_length = data.get("code_length", code_length)
    width, height = data["size"]
    use_diff = data.get("use_diff", False)
    num_codes_r = data.get("num_codes_r")
    num_codes_g = data.get("num_codes_g")
    num_codes_b = data.get("num_codes_b")

    ba_r = data["data_r"]
    ba_g = data["data_g"]
    ba_b = data["data_b"]

    bin_r = "".join(format(byte, '08b') for byte in ba_r)
    bin_g = "".join(format(byte, '08b') for byte in ba_g)
    bin_b = "".join(format(byte, '08b') for byte in ba_b)

    unpadded_r = remove_padding(bin_r)
    unpadded_g = remove_padding(bin_g)
    unpadded_b = remove_padding(bin_b)

    if num_codes_r is not None:
        expected_r = num_codes_r * code_length
        unpadded_r = unpadded_r[:expected_r]
    if num_codes_g is not None:
        expected_g = num_codes_g * code_length
        unpadded_g = unpadded_g[:expected_g]
    if num_codes_b is not None:
        expected_b = num_codes_b * code_length
        unpadded_b = unpadded_b[:expected_b]

    codes_r = binary_string_to_int_array(unpadded_r, code_length)
    codes_g = binary_string_to_int_array(unpadded_g, code_length)
    codes_b = binary_string_to_int_array(unpadded_b, code_length)

    str_r = lzw_decompress(codes_r)
    str_g = lzw_decompress(codes_g)
    str_b = lzw_decompress(codes_b)

    pixels_r = [ord(ch) for ch in str_r]
    pixels_g = [ord(ch) for ch in str_g]
    pixels_b = [ord(ch) for ch in str_b]

    if use_diff:
        pixels_r = reconstruct_pixels(pixels_r, width, height, use_diff)
        pixels_g = reconstruct_pixels(pixels_g, width, height, use_diff)
        pixels_b = reconstruct_pixels(pixels_b, width, height, use_diff)

    expected_pixel_count = width * height
    pixels_r = pixels_r[:expected_pixel_count]
    pixels_g = pixels_g[:expected_pixel_count]
    pixels_b = pixels_b[:expected_pixel_count]

    r_img = Image.new("L", (width, height))
    g_img = Image.new("L", (width, height))
    b_img = Image.new("L", (width, height))
    r_img.putdata(pixels_r)
    g_img.putdata(pixels_g)
    b_img.putdata(pixels_b)

    out_img = Image.merge("RGB", (r_img, g_img, b_img))
    out_img.save(output_image_path)

    return out_img

#Tkinter GUI
class LZWApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LZW Compression / Decompression Program")
        self.filepath = None
        self.use_diff = tk.BooleanVar()          # Difference
        self.output_grayscale = tk.BooleanVar()  # Grayscale
        self.create_widgets()

    def create_widgets(self):
        self.btn_select = tk.Button(self.root, text="Select File", command=self.select_file)
        self.btn_select.pack(pady=5)

        self.lbl_file = tk.Label(self.root, text="(No file selected)")
        self.lbl_file.pack(pady=5)
        
        self.chk_diff = tk.Checkbutton(self.root, text="Use Difference Method (Image)", variable=self.use_diff)
        self.chk_diff.pack(pady=5)
        
        self.chk_gray = tk.Checkbutton(self.root, text="Grayscale Compression", variable=self.output_grayscale)
        self.chk_gray.pack(pady=5)
        
        self.btn_compress = tk.Button(self.root, text="Compress", command=self.compress_file)
        self.btn_compress.pack(pady=5)

        self.btn_decompress = tk.Button(self.root, text="Decompress", command=self.decompress_file)
        self.btn_decompress.pack(pady=5)

        self.btn_metrics = tk.Button(self.root, text="Show Metrics", command=self.show_metrics)
        self.btn_metrics.pack(pady=5)

        self.lbl_info = tk.Label(self.root, text="")
        self.lbl_info.pack(pady=5)

    def select_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("PNG, BMP, or TXT Files", "*.png *.bmp *.txt")])
        if self.filepath:
            self.lbl_file.config(text=self.filepath)
        else:
            self.lbl_file.config(text="(No file selected)")

    def compress_file(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select a file first!")
            return

        filename, ext = os.path.splitext(self.filepath)
        output_path = filename + "_compressed.lzw"

        try:
            if ext.lower() == ".txt":
                compress_text_file(self.filepath, output_path)  
                info = f"Text file compressed.\nCompressed file: {output_path}"
            
            elif ext.lower() in [".png", ".bmp"]:
                if self.output_grayscale.get():
                    compress_gray_image(self.filepath, output_path, use_diff=self.use_diff.get())
                    info = f"Grayscale image compressed.\nCompressed file: {output_path}"
                else:
                    compress_color_image(self.filepath, output_path, use_diff=self.use_diff.get())
                    info = f"Color image compressed.\nCompressed file: {output_path}"
            else:
                info = "Unsupported file type!"
            self.lbl_info.config(text=info)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def decompress_file(self):
        compressed_file = filedialog.askopenfilename(filetypes=[("LZW Files", "*.lzw")])
        if not compressed_file:
            return
        try:
            with open(compressed_file, "rb") as f:
                data = pickle.load(f)

            if data.get("type") == "text":
                output_path = os.path.splitext(compressed_file)[0] + "_decompressed.txt"
                decompress_text_file(compressed_file, output_path)
                info = f"Text file decompressed.\nOutput file: {output_path}"

            elif data.get("type") == "gray":
                output_path = os.path.splitext(compressed_file)[0] + "_decompressed.png"
                decompress_gray_image(compressed_file, output_path)
                info = f"Grayscale image decompressed.\nOutput file: {output_path}"

            elif data.get("type") == "color":
                output_path = os.path.splitext(compressed_file)[0] + "_decompressed.png"
                decompress_color_image(compressed_file, output_path)
                info = f"Color image decompressed.\nOutput file: {output_path}"
            else:
                info = "Unknown file type!"
            self.lbl_info.config(text=info)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_metrics(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select a file first!")
            return
        
        compressed_file = filedialog.askopenfilename(filetypes=[("LZW Files", "*.lzw")])

        if not compressed_file:
            return

        try:
            metrics = show_compression_metrics(self.filepath, compressed_file)
            messagebox.showinfo("Compression Metrics", metrics)
        except Exception as e:
            messagebox.showerror("Error", str(e))

#Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = LZWApp(root)
    root.mainloop()
