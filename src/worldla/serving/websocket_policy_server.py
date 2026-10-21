import asyncio
import logging
import traceback
import cv2  # 【新增】需要 cv2 进行解码
import numpy as np # 【新增】需要 numpy
import base64
from worldla.client import interface as _base_policy
from worldla.client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.PolicyInterface,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # 保持禁用 Ping 的设置
        async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = await websocket.recv()
                
                # 1. 基础解包
                if isinstance(msg, str):
                    msg = msg.encode('latin-1')
                obs = msgpack_numpy.unpackb(msg)

                # 2. 【核心】Base64 解码 + 维度转换
                if isinstance(obs, dict) and "images" in obs:
                    for key, val in obs["images"].items():
                        # 检查是否是字符串 (Base64)
                        if isinstance(val, str):
                            try:
                                # A. Base64 -> Bytes
                                image_bytes = base64.b64decode(val)
                                
                                # B. Bytes -> Numpy (H, W, C)
                                nparr = np.frombuffer(image_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if img is not None:
                                    # C. BGR -> RGB
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    
                                    # D. 【关键修正】 HWC -> CHW (3, 224, 224)
                                    # 因为 aloha_policy.py 期望输入是 Channel First
                                    img = img.transpose(2, 0, 1)
                                    
                                    obs["images"][key] = img
                                    # print(f"DEBUG: Decoded {key} shape: {img.shape}") 
                                else:
                                    logging.error(f"Failed to decode image: {key}")
                            except Exception as e:
                                logging.error(f"Error processing image {key}: {e}")
                        
                        # 如果是 bytes (Raw JPEG)，也做同样的逻辑
                        elif isinstance(val, bytes):
                            try:
                                nparr = np.frombuffer(val, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = img.transpose(2, 0, 1) # HWC -> CHW
                                    obs["images"][key] = img
                            except Exception as e:
                                logging.error(f"Error processing raw bytes {key}: {e}")

                # 3. 推理
                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # 打印详细错误堆栈到服务器控制台
                traceback.print_exc() 
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise