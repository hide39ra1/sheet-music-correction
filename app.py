import io
import json
import math
import re
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


def corrected(image, angle, flip, fit):
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


def load_pdf(source, folder):
    info = poppler("pdfinfo", str(source)).decode(errors="replace")
    count = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    info = poppler("pdfinfo", "-f", "1", "-l", str(count), str(source)).decode(errors="replace")
    sizes = re.findall(r"Page\s+\d+\s+size:\s+([\d.]+) x ([\d.]+)", info)
    rotations = re.findall(r"Page\s+\d+\s+rot:\s+(\d+)", info)
    if len(sizes) != count:
        raise ValueError("PDFのページ寸法を取得できませんでした")
    poppler("pdftoppm", "-cropbox", "-r", "300", "-png", str(source), str(folder / "page"))
    paths = sorted(folder.glob("page-*.png"))
    pages = []
    for index, (path, (w, h)) in enumerate(zip(paths, sizes)):
        w, h = float(w), float(h)
        if rotations and int(rotations[index]) % 180:
            w, h = h, w
        pages.append((path, (w, h)))
    if len(pages) != count:
        raise ValueError("PDFの一部を読み込めませんでした")
    return pages


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
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            self.reply(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

    def process(self):
        import base64
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 512 * 1024**2:
            raise ValueError("入力サイズが上限の512 MBを超えています")
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
            pages = load_pdf(source, folder) if pdf else [(source, None)]
            if self.path == "/analyse":
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
                settings = json.loads(self.headers["X-Settings"])
                if len(settings) != len(pages):
                    raise ValueError("ページ設定の数が一致しません")
                def images():
                    for (path, dimensions), (angle, flip, fit) in zip(pages, settings):
                        angle = float(angle)
                        if not math.isfinite(angle) or abs(angle) > 15:
                            raise ValueError("角度は±15°以内で指定してください")
                        yield corrected(rgb(path), angle, flip, fit), dimensions
                output = io.BytesIO()
                if pdf:
                    write_pdf(output, _PageStream(len(pages), images()))
                else:
                    image, _ = next(images())
                    image.save(output, FORMATS[fmt], **({"quality": 95} if fmt in (".jpg", ".jpeg", ".webp") else {}))
                self.reply(output.getvalue(), "application/pdf" if pdf else "application/octet-stream")
            else:
                self.send_error(404)


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
