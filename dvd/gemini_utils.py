import os
import time
import random
from typing import Optional

from google import genai
from google.genai import types


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "limit" in error_str or "timeout" in error_str or "503" in error_str or "500" in error_str:
                    num_retries += 1
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay:.1f} seconds for {str(e)}...")
                    time.sleep(delay)
                else:
                    print(f"Gemini API Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None

    return wrapper


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


@retry_with_exponential_backoff
def call_gemini_with_video(
    video_path: str,
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    start_offset_sec: Optional[float] = None,
    end_offset_sec: Optional[float] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[str]:
    client = get_gemini_client(api_key)
    
    uploaded_file = client.files.upload(file=video_path)
    
    while uploaded_file.state == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    
    if uploaded_file.state == "FAILED":
        raise ValueError(f"Video processing failed: {uploaded_file.state}")
    
    contents = []
    
    if start_offset_sec is not None and end_offset_sec is not None:
        video_part = types.Part.from_uri(
            file_uri=uploaded_file.uri,
            mime_type=uploaded_file.mime_type,
        )
        contents.append(video_part)
        contents.append(f"Focus on the video segment from {start_offset_sec:.1f}s to {end_offset_sec:.1f}s. {prompt}")
    else:
        contents.append(uploaded_file)
        contents.append(prompt)
    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass
    
    return response.text


@retry_with_exponential_backoff
def call_gemini_with_video_clip(
    video_path: str,
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[str]:
    client = get_gemini_client(api_key)
    
    uploaded_file = client.files.upload(file=video_path)
    
    while uploaded_file.state == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    
    if uploaded_file.state == "FAILED":
        raise ValueError(f"Video processing failed: {uploaded_file.state}")
    
    response = client.models.generate_content(
        model=model_name,
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass
    
    return response.text


@retry_with_exponential_backoff
def call_gemini_with_frames(
    image_paths: list[str],
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[str]:
    client = get_gemini_client(api_key)
    
    contents = []
    
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        
        mime_type = "image/jpeg" if img_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    
    contents.append(prompt)
    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    
    return response.text


@retry_with_exponential_backoff
def call_gemini_with_youtube(
    youtube_url: str,
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[str]:
    client = get_gemini_client(api_key)
    
    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=youtube_url)
                ),
                types.Part(text=prompt)
            ]
        ),
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    
    return response.text


@retry_with_exponential_backoff
def call_gemini_text(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    return_json: bool = False,
) -> Optional[str]:
    client = get_gemini_client(api_key)
    
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    if return_json:
        config.response_mime_type = "application/json"
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    
    return response.text
