#!/usr/bin/env python3
"""
LZW Compression and Decompression with GUI
---------------------------------------------------

Bu program, LZW algoritmasını kullanarak bir resmi (PNG veya BMP) veya metin dosyasını sıkıştırır,
sıkıştırılmış dosyayı kaydeder ve geri açarak orijinal içeriğe ulaşır. Metin ve grayscale görüntüler
için bit seviyesinde paketleme yapar. Orijinal resmin boyutları da sıkıştırılan veriye kaydedilir.
'Force Grayscale Output' seçiliyse, sıkıştırma aşamasında görüntü direkt olarak siyah-beyaza
çevrilip *grayscale* kanaldan sıkıştırılır (renkli bir görüntüyü önce renkli saklayıp sonradan
L'ye çevirmek yerine doğrudan gri kanal olarak okur ve LZW uygular).

Gerekli kütüphaneler:
  - Pillow (PIL): pip install Pillow
  - tkinter (Python'un standart kütüphanesi)
  - pickle (Python'un standart kütüphanesi)
  - math (standart kütüphane)
"""

import os
import pickle
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

# -----------------------------------
# Binary Packaging Functions (for text and grayscale images)
# -----------------------------------
def int_array_to_binary_string(int_array, code_length):
    """Belirlenmiş code_length bit kullanarak, tamsayı listesini bit string'e çevirir."""
    return "".join(format(num, f'0{code_length}b') for num in int_array)

def pad_encoded_text(encoded_text):
    """Bit uzunluğu 8'in katı olacak şekilde pad'leme yapar.
       Başına 8 bitlik padding miktarı bilgisi ekler."""
    extra_padding = (8 - len(encoded_text) % 8) % 8
    padded_info = format(extra_padding, '08b')
    padded_text = padded_info + encoded_text + ("0" * extra_padding)
    return padded_text

def get_byte_array(padded_encoded_text):
    """Pad'lenmiş bit dizisini bytearray'e dönüştürür."""
    return bytearray(int(padded_encoded_text[i:i+8], 2) for i in range(0, len(padded_encoded_text), 8))

def remove_padding(padded_encoded_text):
    """Başta yer alan 8 bitlik padding bilgisini okuyarak orijinal bit dizisini döndürür."""
    extra_padding = int(padded_encoded_text[:8], 2)
    return padded_encoded_text[8:-extra_padding] if extra_padding != 0 else padded_encoded_text[8:]

def binary_string_to_int_array(bitstr, code_length):
    """Sabit code_length bitlik parçalara bölerek bit string'i tamsayı dizisine dönüştürür."""
    return [int(bitstr[i:i+code_length], 2) for i in range(0, len(bitstr), code_length)]

# -----------------------------------
# Compression Metrics Functions
# -----------------------------------
def calculate_compression_metrics(original_path, compressed_path):
    """Orijinal ve sıkıştırılmış dosya boyutu ve oranını hesaplar."""
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    ratio = compressed_size / original_size if original_size else 0
    return original_size, compressed_size, ratio

def calculate_entropy(data):
    """Verilen string'in entropisini (bit/simge) hesaplar."""
    freq = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(data)
    return -sum((count/total) * math.log2(count/total) for count in freq.values())

def show_compression_metrics(original_path, compressed_path):
    """Dosya boyutları ve entropiyi içeren bir metin döndürür."""
    orig_size, comp_size, ratio = calculate_compression_metrics(original_path, compressed_path)
    with open(original_path, "rb") as f:
        data = f.read()
    # Metin dosyaları için UTF-8 deneyelim, hata olursa hex'e çevirip entropi hesaplıyoruz
    try:
        data = data.decode("utf-8")
    except UnicodeDecodeError:
        data = data.hex()
    entropy_val = calculate_entropy(data)
    return (f"Original Size: {orig_size} bytes\n"
            f"Compressed Size: {comp_size} bytes\n"
            f"Compression Ratio: {ratio:.2f}\n"
            f"Entropy: {entropy_val:.2f} bits/symbol")

# -----------------------------------
# LZW Compression / Decompression Functions
# -----------------------------------
def lzw_compress(uncompressed):
    """Verilen string'i LZW algoritmasıyla sıkıştırır, integer kod listesi döndürür."""
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
    """LZW kod listesini alıp orijinal string'i döndürür."""
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

# -----------------------------------
# Text File Compression & Decompression (with Bit-Level Packaging)
# -----------------------------------
def compress_text_file(filepath, output_path, code_length=12):
    """Bir metin dosyasını LZW ve bit-level paketleme yaparak sıkıştırır, pickle ile kaydeder."""
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

def decompress_text_file(filepath, output_txt_path, code_length=12):
    """Sıkıştırılmış metin dosyasını okur, açar ve bir .txt dosyasına yazar."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "text":
        raise ValueError("Bu dosya metin sıkıştırması içermiyor!")

    byte_array = data["data"]
    bit_str = "".join(format(byte, '08b') for byte in byte_array)
    unpadded = remove_padding(bit_str)
    codes = binary_string_to_int_array(unpadded, code_length)
    text = lzw_decompress(codes)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_txt_path

# -----------------------------------
# Image Pixel Processing Functions
# -----------------------------------
def get_image_pixels(img, use_diff=False):
    """Verilen PIL image objesinden piksel verilerini çıkarır.
       use_diff=True ise, difference methodu uygulanır."""
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
    """Difference verileri kullanılarak orijinal piksel değerlerini geri dönüştürür."""
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

# -----------------------------------
# Grayscale Image Compression & Decompression (with Bit-Level Packaging)
# -----------------------------------
def compress_gray_image(filepath, output_path, use_diff=False, code_length=12):
    """
    Girilen resmi doğrudan 'L' (grayscale) modda açar (veya dönüştürür).
    Piksel değerlerini LZW ile bit-level paketlenmiş biçimde sıkıştırır ve kaydeder.
    """
    img = Image.open(filepath).convert("L")
    pixels, width, height = get_image_pixels(img, use_diff)
    pixel_string = "".join(chr(p) for p in pixels)
    codes = lzw_compress(pixel_string)
    bit_string = int_array_to_binary_string(codes, code_length)
    padded = pad_encoded_text(bit_string)
    byte_array = get_byte_array(padded)

    data = {
        "type": "image",  # Basitçe 'image' diyebiliriz
        "mode": "L",
        "size": img.size,
        "use_diff": use_diff,
        "code_length": code_length,
        "num_codes": len(codes),
        "data": byte_array
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return output_path

def decompress_gray_image(filepath, output_image_path, code_length=12):
    """Sıkıştırılmış gri görüntüyü açıp PNG formatında kaydeder."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "image":
        raise ValueError("Bu dosyada grayscale görüntü sıkıştırması yok!")

    size = data["size"]
    use_diff = data.get("use_diff", False)
    byte_array = data["data"]
    bit_str = "".join(format(byte, '08b') for byte in byte_array)
    unpadded = remove_padding(bit_str)
    codes = binary_string_to_int_array(unpadded, code_length)
    decompressed_string = lzw_decompress(codes)
    pixels = [ord(c) for c in decompressed_string]
    img = Image.new("L", size)
    img.putdata(pixels)
    img.save(output_image_path)

    return output_image_path

# -----------------------------------
# Color Image Compression & Decompression (Using Pickle Storage)
# -----------------------------------
def compress_color_image(filepath, output_path, use_diff=False, code_length=12):
    """
    Renkli resmi her bir R, G, B kanalı ayrı ayrı olacak şekilde LZW ile sıkıştırır.
    İsteğe bağlı difference coding de eklenebilir (use_diff=True).
    """
    img = Image.open(filepath).convert("RGB")
    width, height = img.size
    r_img, g_img, b_img = img.split()

    # Gerekirse difference methodu
    if use_diff:
        pixels_r, _, _ = get_image_pixels(r_img, use_diff)
        pixels_g, _, _ = get_image_pixels(g_img, use_diff)
        pixels_b, _, _ = get_image_pixels(b_img, use_diff)
    else:
        pixels_r = list(r_img.getdata())
        pixels_g = list(g_img.getdata())
        pixels_b = list(b_img.getdata())

    # Her kanal 0-255 arası karakter dizisine dönüşür
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
        "mode": "RGB",
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

def decompress_color_image(filepath, output_image_path, code_length=12):
    """
    Renkli görüntüyü açıp (her kanalı çözüp) yeniden birleştirerek RGB görseli oluşturur.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if data.get("type") != "color":
        raise ValueError("Bu dosyada color image sıkıştırması yok!")

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

# -----------------------------------
# Tkinter GUI
# -----------------------------------
class LZWApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LZW Compression / Decompression Program")
        self.filepath = None
        self.use_diff = tk.BooleanVar()          # Difference Method
        self.output_grayscale = tk.BooleanVar()  # Force Grayscale
        self.create_widgets()

    def create_widgets(self):
        self.btn_select = tk.Button(self.root, text="Select File", command=self.select_file)
        self.btn_select.pack(pady=5)

        self.lbl_file = tk.Label(self.root, text="(No file selected)")
        self.lbl_file.pack(pady=5)
        
        self.chk_diff = tk.Checkbutton(self.root, text="Use Difference Method (Image)", variable=self.use_diff)
        self.chk_diff.pack(pady=5)
        
        self.chk_gray = tk.Checkbutton(self.root, text="Force Grayscale Output", variable=self.output_grayscale)
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
        # PNG, BMP, TXT formatlarına izin ver
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
                # Metin dosyası
                compress_text_file(self.filepath, output_path)
                info = f"Text file compressed.\nCompressed file: {output_path}"
            
            elif ext.lower() in [".png", ".bmp"]:
                # Resim dosyası
                # Force Grayscale seçiliyse, doğrudan grayscale olarak sıkıştır
                if self.output_grayscale.get():
                    compress_gray_image(self.filepath, output_path, use_diff=self.use_diff.get())
                    info = f"Image compressed as GRAYSCALE (forced).\nCompressed file: {output_path}"
                else:
                    # Eğer resmin mode'u L ise zaten grayscale
                    img = Image.open(self.filepath)
                    if img.mode == "L":
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
                # Grayscale
                output_path = os.path.splitext(compressed_file)[0] + "_decompressed.png"
                decompress_gray_image(compressed_file, output_path)
                info = f"Grayscale image decompressed.\nOutput file: {output_path}"

            elif data.get("type") == "color":
                # Renkli
                output_path = os.path.splitext(compressed_file)[0] + "_decompressed.png"
                img = decompress_color_image(compressed_file, output_path)
                # Önceden color->L çeviriyorduk. Artık 'Force Grayscale' burada
                # sadece orijinal color veriden sonradan L yapıyor. Ama
                # eğer "Force Grayscale" i bu aşamada da istiyorsanız
                # *sıkıştırma* safhasında değil, *sonuç* safhasında L'e döner.
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

# -----------------------------------
# Main Program
# -----------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LZWApp(root)
    root.mainloop()
