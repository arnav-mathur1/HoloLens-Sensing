import socket
import struct
import numpy as np
import cv2

HOST = "0.0.0.0"
PORT = 23947

HEADER_FORMAT = "<IBBHQIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

FRAME_COOKIE = 0x484C524D

MATRIX_FORMAT = "<16f"
MATRIX_SIZE = struct.calcsize(MATRIX_FORMAT)

CALIB_SIZE_FORMAT = "<I"
CALIB_SIZE_SIZE = struct.calcsize(CALIB_SIZE_FORMAT)

SAVE_FRAME = 15
STOP_FRAME = 25


def recv_exact(conn, num_bytes):
    data = bytearray()

    while len(data) < num_bytes:
        chunk = conn.recv(num_bytes - len(data))

        if not chunk:
            raise ConnectionError("HoloLens disconnected")

        data.extend(chunk)

    return bytes(data)


def read_matrix(conn):
    data = recv_exact(conn, MATRIX_SIZE)

    return np.array(
        struct.unpack(MATRIX_FORMAT, data),
        dtype=np.float32
    ).reshape(4, 4)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)

print("Waiting for HoloLens on port", PORT)

conn, addr = server.accept()
print("Connected from:", addr)

frame_number = 0
ray_lut = None

try:
    while True:

        # --------------------------------------------------
        # Header
        # --------------------------------------------------

        header_data = recv_exact(conn, HEADER_SIZE)

        (
            cookie,
            version_major,
            version_minor,
            frame_type,
            timestamp,
            width,
            height,
            pixel_stride,
            row_stride
        ) = struct.unpack(
            HEADER_FORMAT,
            header_data
        )

        if cookie != FRAME_COOKIE:
            print("BAD COOKIE:", hex(cookie))
            break

        # --------------------------------------------------
        # FrameToOrigin
        # --------------------------------------------------

        frame_to_origin = read_matrix(conn)

        # --------------------------------------------------
        # CameraViewTransform
        # --------------------------------------------------

        camera_view_transform = read_matrix(conn)

        # --------------------------------------------------
        # Calibration LUT size
        # --------------------------------------------------

        calibration_bytes = struct.unpack(
            CALIB_SIZE_FORMAT,
            recv_exact(conn, CALIB_SIZE_SIZE)
        )[0]

        # --------------------------------------------------
        # Calibration LUT
        # --------------------------------------------------

        if calibration_bytes > 0:

            calibration_data = recv_exact(
                conn,
                calibration_bytes
            )

            calibration_values = np.frombuffer(
                calibration_data,
                dtype="<f4"
            )

            expected = width * height * 3

            if len(calibration_values) != expected:
                raise RuntimeError(
                    f"Calibration mismatch: "
                    f"expected {expected}, "
                    f"got {len(calibration_values)}"
                )

            ray_lut = calibration_values.reshape(
                height,
                width,
                3
            ).copy()

            np.save(
                "longthrow_lut.npy",
                ray_lut
            )

            print(
                "Saved calibration LUT:",
                ray_lut.shape
            )

        # --------------------------------------------------
        # Depth image
        # --------------------------------------------------

        image_size = row_stride * height

        image_data = recv_exact(
            conn,
            image_size
        )

        frame_number += 1

        if pixel_stride != 2:
            print(
                "Unexpected pixel stride:",
                pixel_stride
            )
            continue

        depth = np.frombuffer(
            image_data,
            dtype="<u2"
        )

        if row_stride == width * 2:

            depth = depth.reshape(
                height,
                width
            )

        else:

            depth = depth.reshape(
                height,
                row_stride // 2
            )[:, :width]

        print(
            f"Frame {frame_number}/{STOP_FRAME}"
        )

        # --------------------------------------------------
        # Show live depth
        # --------------------------------------------------

        depth_vis = np.clip(
            depth,
            0,
            4000
        )

        depth_vis = (
            depth_vis / 4000.0 * 255
        ).astype(np.uint8)

        cv2.imshow(
            "HoloLens LongThrow Depth",
            depth_vis
        )

        cv2.waitKey(1)

        # --------------------------------------------------
        # Save frame 15
        # --------------------------------------------------

        if frame_number == SAVE_FRAME:

            np.save(
                "depth_frame_15.npy",
                depth.copy()
            )

            np.save(
                "frame_to_origin_15.npy",
                frame_to_origin.copy()
            )

            np.save(
                "camera_view_transform_15.npy",
                camera_view_transform.copy()
            )

            cv2.imwrite(
                "depth_frame_15_preview.png",
                depth_vis
            )

            print()
            print("SAVED FRAME 15")
            print("  depth_frame_15.npy")
            print("  depth_frame_15_preview.png")
            print("  frame_to_origin_15.npy")
            print("  camera_view_transform_15.npy")
            print("  longthrow_lut.npy")
            print()

        # --------------------------------------------------
        # Stop after frame 25
        # --------------------------------------------------

        if frame_number >= STOP_FRAME:

            print(
                "Reached frame 25. Stopping."
            )

            break


except ConnectionError as e:
    print(e)

except KeyboardInterrupt:
    print("\nStopped manually.")

finally:

    conn.close()
    server.close()

    cv2.destroyAllWindows()

    print("Receiver closed.")