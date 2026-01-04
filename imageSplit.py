from PIL import Image
import requests
from io import BytesIO

def split_image(image_url, proxy = "socks5://xxx", grid_size=(3, 3)):
    try:

        proxies = {
            'http': proxy,
            'https': proxy
        }

        response = requests.get(
                        "https://image-fetcher-766107398946.us-central1.run.app/fetch?url=" + image_url,
                        #proxies=proxies,
                        verify=False,
                        timeout=180
                    )
        image = Image.open(BytesIO(response.content))
        img_width, img_height = image.size
        grid_w, grid_h = grid_size
        tile_width, tile_height = img_width // grid_w, img_height // grid_h
        
        images = []
        for row in range(grid_h):
            for col in range(grid_w):
                left = col * tile_width
                upper = row * tile_height
                right = left + tile_width
                lower = upper + tile_height
                cropped_img = image.crop((left, upper, right, lower))
                images.append(cropped_img)
        
        return { "status": True, "data": images }
    except Exception as e:
        return { "status": False, "error": str(e) }
