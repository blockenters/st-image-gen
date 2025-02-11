import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io

st.title("🎨 AI 이미지 생성기")
st.write("텍스트를 입력하면 AI가 이미지를 생성합니다!")

# 모델 로드 (세션 상태로 관리)
@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# 사용자 입력 받기
prompt = st.text_input("이미지로 만들고 싶은 내용을 영어로 적어주세요:", 
                      "A fantasy landscape with castles and dragons")

if st.button("이미지 생성하기"):
    with st.spinner("이미지를 생성하는 중..."):
        # 이미지 생성
        image = pipe(prompt).images[0]
        
        # 이미지 표시
        st.image(image, caption="생성된 이미지")
        
        # 이미지 다운로드 버튼
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        btn = st.download_button(
            label="이미지 다운로드",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )

