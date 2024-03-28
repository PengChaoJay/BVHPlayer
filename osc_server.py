import asyncio
import websockets

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.first_message_received = False

    async def start_server(self):
        start_server = websockets.serve(self.server_handler, self.host, self.port)
        await start_server

    async def server_handler(self, websocket, path):
        try:
            while True:
                message = await websocket.recv()
                print(f"Received message: {message}")
                if not self.first_message_received:
                    await self.send_message(websocket, "First message received.")
                    self.first_message_received = True
        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected")
            self.first_message_received = False

    async def send_message(self, websocket, message):
        await websocket.send(message)

if __name__ == "__main__":
    server = WebSocketServer("localhost", 8888)
    asyncio.get_event_loop().run_until_complete(server.start_server())
    asyncio.get_event_loop().run_forever()
