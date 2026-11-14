import io
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image

FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
           ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP", ".webp": "WEBP"}


def detect(image):
    gray = np.asarray(image.convert("L"))
    scale = min(1, 1200 / max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 60, 160)
    width = gray.shape[1]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 65,
                            minLineLength=max(35, width // 6), maxLineGap=16)
    if lines is None:
        return None
    votes = {}
    for x, y, xx, yy in lines[:, 0]:
        if x == xx:
            continue
        angle = math.degrees(math.atan((yy - y) / (xx - x)))
        if abs(angle) <= 10:
            votes.setdefault(round(angle * 4), []).append((angle, abs(xx - x)))
    if not votes:
        return None
    peak = max(votes, key=lambda k: sum(w for j in range(k-1, k+2)
                                        for _, w in votes.get(j, [])))
    group = [pair for j in range(peak-1, peak+2) for pair in votes.get(j, [])]
    if len(group) < 5:
        return None
    return round(sum(a*w for a, w in group) / sum(w for _, w in group), 2)


def corrected(image, angle, flip, fit, margins=(0, 0, 0, 0)):
    if len(margins) != 4 or any(not 0 <= value <= 20 for value in margins):
        raise ValueError("端の白塗りは各辺0〜20%で指定してください")
    top, right, bottom, left = (
        round(value * dimension / 100)
        for value, dimension in zip(margins, (image.height, image.width, image.height, image.width))
    )
    width, height = image.size
    for box in ((0, 0, width, top), (width-right, 0, width, height),
                (0, height-bottom, width, height), (0, 0, left, height)):
        image.paste("white", box)
    if flip:
        image = image.transpose(Image.Transpose.ROTATE_180)
    if not angle:
        return image
    rotated = image.rotate(angle, Image.Resampling.BICUBIC, expand=fit, fillcolor="white")
    if not fit:
        return rotated
    rotated.thumbnail(image.size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(rotated, ((image.width-rotated.width)//2, (image.height-rotated.height)//2))
    return canvas


def poppler(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, timeout=300)


def pdf_dimensions(source):
    count = pdf_page_count(source)
    info = poppler("pdfinfo", "-f", "1", "-l", str(count), str(source)).decode(errors="replace")
    sizes = re.findall(r"Page\s+\d+\s+size:\s+([\d.]+) x ([\d.]+)", info)
    rotations = re.findall(r"Page\s+\d+\s+rot:\s+(\d+)", info)
    if len(sizes) != count:
        raise ValueError("PDFのページ寸法を取得できませんでした")
    dimensions = []
    for index, (w, h) in enumerate(sizes):
        w, h = float(w), float(h)
        if rotations and int(rotations[index]) % 180:
            w, h = h, w
        dimensions.append((w, h))
    return dimensions


def pdf_pages(source, folder, dimensions):
    batch_size = 8
    for start in range(0, len(dimensions), batch_size):
        batch = dimensions[start:start + batch_size]
        poppler("pdftoppm", "-f", str(start + 1), "-l", str(start + len(batch)),
                "-cropbox", "-r", "300", "-png", str(source), str(folder / "page"))
        paths = sorted(folder.glob("page-*.png"))
        if len(paths) != len(batch):
            raise ValueError("PDFの一部を読み込めませんでした")
        for path, size in zip(paths, batch):
            yield path, size
            path.unlink()


def pdf_page_count(source):
    info = poppler("pdfinfo", str(source)).decode(errors="replace")
    match = re.search(r"^Pages:\s+(\d+)", info, re.M)
    if not match:
        raise ValueError("PDFのページ数を取得できませんでした")
    return int(match[1])


def write_pdf(output, pages):
    offsets = [0]
    output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def obj(payload):
        offsets.append(output.tell())
        output.write(f"{len(offsets)-1} 0 obj\n".encode() + payload + b"\nendobj\n")

    def stream(header, data):
        obj(header + f" /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")

    obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3+3*i} 0 R" for i in range(len(pages)))
    obj(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode())
    for index, (image, (w, h)) in enumerate(pages):
        base = 3 + 3*index
        obj(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /XObject << /Scan {base+1} 0 R >> >> /Contents {base+2} 0 R >>".encode())
        jpeg = io.BytesIO()
        image.save(jpeg, "JPEG", quality=95, subsampling=0)
        stream(f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode".encode(), jpeg.getvalue())
        stream(b"<<", f"q {w} 0 0 {h} 0 0 cm /Scan Do Q".encode())
    start = output.tell()
    output.write(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())


def rgb(path):
    with Image.open(path) as image:
        if image.n_frames != 1:
            raise ValueError("複数フレームの画像には対応していません")
        rgba = image.convert("RGBA")
        result = Image.new("RGB", image.size, "white")
        result.paste(rgba, mask=rgba.getchannel("A"))
        return result


class Server(BaseHTTPRequestHandler):
    def reply(self, data, kind, status=200):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.headers.get("Host") != "127.0.0.1:8765" or self.headers.get("Origin") != "http://127.0.0.1:8765":
            self.send_error(403)
            return
        try:
            self.process()
        except ConnectionError:
            return
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            try:
                self.reply(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
            except ConnectionError:
                return

    def process(self):
        import base64
        size = int(self.headers.get("Content-Length", "0"))
        settings_length = int(self.headers.get("X-Settings-Length", "0"))
        if not 0 <= settings_length <= min(8 * 1024**2, size):
            raise ValueError("ページ設定のサイズが無効です")
        size -= settings_length
        if not 0 < size <= 512 * 1024**2:
            raise ValueError("入力サイズが上限の512 MBを超えています")
        settings_data = self.headers.get("X-Settings", "[]")
        if settings_length:
            settings_data = self.rfile.read(settings_length)
            if len(settings_data) != settings_length:
                raise ValueError("ページ設定の受信が途中で終了しました")
        with tempfile.TemporaryDirectory(prefix="score-") as temp:
            folder = Path(temp)
            source = folder / "input"
            with source.open("wb") as handle:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024**2))
                    if not chunk:
                        raise ValueError("受信が途中で終了しました")
                    handle.write(chunk)
                    remaining -= len(chunk)
            fmt = self.headers.get("X-Format", "").lower()
            pdf = fmt == ".pdf"
            if not pdf and fmt not in FORMATS:
                raise ValueError("未対応の形式です")
            if self.path == "/analyse":
                if self.headers.get("Accept") == "application/x-ndjson":
                    self.stream_analysis(source, folder, pdf)
                    return
                pages = pdf_pages(source, folder, pdf_dimensions(source)) if pdf else [(source, None)]
                previews = []
                for path, _ in pages:
                    image = rgb(path)
                    angle = detect(image)
                    image.thumbnail((800, 1050))
                    preview = io.BytesIO()
                    image.save(preview, "JPEG", quality=85)
                    previews.append({"angle": angle, "preview": base64.b64encode(preview.getvalue()).decode()})
                self.reply(json.dumps(previews).encode(), "application/json")
            elif self.path == "/export":
                dimensions = pdf_dimensions(source) if pdf else [None]
                pages = pdf_pages(source, folder, dimensions) if pdf else [(source, None)]
                settings = json.loads(settings_data)
                if len(settings) != len(dimensions):
                    raise ValueError("ページ設定の数が一致しません")
                def images():
                    for (path, dimensions), (angle, flip, fit, *options) in zip(pages, settings):
                        angle = float(angle)
                        if not math.isfinite(angle) or abs(angle) > 15:
                            raise ValueError("角度は±15°以内で指定してください")
                        yield corrected(rgb(path), angle, flip, fit, *options), dimensions
                if pdf:
                    with tempfile.TemporaryFile(dir=folder) as output:
                        write_pdf(output, _PageStream(len(dimensions), images()))
                        length = output.tell()
                        output.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/pdf")
                        self.send_header("Content-Length", str(length))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        shutil.copyfileobj(output, self.wfile, length=256 * 1024)
                else:
                    output = io.BytesIO()
                    image, _ = next(images())
                    image.save(output, FORMATS[fmt], **({"quality": 95} if fmt in (".jpg", ".jpeg", ".webp") else {}))
                    self.reply(output.getvalue(), "application/octet-stream")
            else:
                self.send_error(404)

    def stream_analysis(self, source, folder, pdf):
        import base64
        count = pdf_page_count(source) if pdf else 1
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(record):
            self.wfile.write((json.dumps(record) + "\n").encode())
            self.wfile.flush()

        try:
            send({"total": count})
            for number in range(1, count + 1):
                path = source
                if pdf:
                    prefix = folder / "preview"
                    poppler("pdftoppm", "-f", str(number), "-l", str(number),
                            "-singlefile", "-cropbox", "-scale-to", "1400", "-png",
                            str(source), str(prefix))
                    path = prefix.with_suffix(".png")
                image = rgb(path)
                angle = detect(image)
                image.thumbnail((800, 1050))
                preview = io.BytesIO()
                image.save(preview, "JPEG", quality=85)
                send({"angle": angle, "preview": base64.b64encode(preview.getvalue()).decode()})
        except ConnectionError:
            return
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            send({"error": str(exc)})


class _PageStream:
    def __init__(self, count, images):
        self.count = count
        self.images = images

    def __len__(self):
        return self.count

    def __iter__(self):
        return self.images


if __name__ == "__main__":
    print("http://127.0.0.1:8765", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", 8765), Server) as server:
        server.serve_forever()
