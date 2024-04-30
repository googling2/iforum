import vPlayer from "./vPlayer.js";

const videoList = [
  {
    url: "../vids/sample1.mp4",
    title: "이준희 작가님",
    desc: "나는 돈이 좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample2.mp4",
    title: "김성민작가님",
    desc: "나는 글을 잘써!",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample3.mp4",
    title: "백종환 작가님",
    desc: "나는 성경이좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample4.mp4",
    title: "이준희 작가님",
    desc: "나는 돈이 좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample5.mp4",
    title: "김성민작가님",
    desc: "나는 글을 잘써!",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample6.mp4",
    title: "백종환 작가님",
    desc: "나는 성경이좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample7.mp4",
    title: "이준희 작가님",
    desc: "나는 돈이 좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample8.mp4",
    title: "김성민작가님",
    desc: "나는 글을 잘써!",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  {
    url: "../vids/sample9.mp4",
    title: "백종환 작가님",
    desc: "나는 성경이좋아",
    img: "https://picsum.photos/200",
    preload: "metadata",
    poster: "https://picsum.photos/200"
  },
  // 나머지 비디오 항목들도 같은 방식으로 수정
];

const videosWrapper = document.querySelector(".videos");
const loader = document.querySelector(".loader");

setTimeout(() => {
  injectVideos();
  loader.style.display = "none";
}, 2000);

function injectVideos() {
  videoList.forEach((video) => {
    videosWrapper.innerHTML += vPlayer(video, video.preload, video.poster);
  });

  const vids = document.querySelectorAll(".vdo-wrapper");
  vids.forEach((vid) => {
    const vdo = vid.querySelector(".vdo");

    // 화면 내비켜진 경우에만 비디오를 재생합니다.
    const intersectionObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.intersectionRatio > 0) {
          vdo.play();
        } else {
          vdo.pause();
        }
      });
    });

    intersectionObserver.observe(vdo);

    const pausePlayBtn = vid.querySelector("#pause-play-btn");
    pausePlayBtn.addEventListener("click", () => {
      if (vdo.paused) {
        vdo.play();
        pausePlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
      } else {
        vdo.pause();
        pausePlayBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
      }
    });

    const likeBtn = vid.querySelector("#like-btn");
    likeBtn.addEventListener("click", () => {
      likeBtn.classList.toggle("color-switch");
      const dislikeBtn = vid.querySelector("#dislike-btn");
      dislikeBtn.classList.remove("color-switch");
    });

    const dislikeBtn = vid.querySelector("#dislike-btn");
    dislikeBtn.addEventListener("click", () => {
      dislikeBtn.classList.toggle("color-switch");
      const likeBtn = vid.querySelector("#like-btn");
      likeBtn.classList.remove("color-switch");
    });

    const commentsBtn = vid.querySelector("#comments-btn");
    const comments = vid.querySelector(".comments");
    commentsBtn.addEventListener("click", () => {
      comments.classList.toggle("toggle-comments");
    });

    const shareBtn = vid.querySelector("#share-btn");
    const vidTitle = vid.querySelector(".info-wrapper h3");
    shareBtn.addEventListener("click", () => {
      navigator.share({
        title: vidTitle.textContent,
        text: "Check out this video!",
        url: vdo.src,
      });
    });
  });
}