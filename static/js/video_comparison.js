var comparisonMethod = "StochasticSplats";
var comparisonScene = "Garden";
// Each scene loads static/video/<Scene>.mp4.
var comparisonScenes = ["Garden", "Bicycle", "Kitchen", "Room", "Stump", "Counter", "Bonsai"];
// Row of each method in the vertically stacked scene videos, top to bottom.
var stackedMethodOrder = {
    Ours: 0,
    StochasticSplats: 1,
    "Gaussian Point Splatting": 2,
    "Sorted 3DGS": 3
};
// Loads the video of comparisonScene into the comparison; set up by setupComparison().
var loadComparisonScene = function() {};

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function getStackedSceneVideoPath(scene) {
    return "static/video/" + scene + ".mp4";
}

// Highlights the button whose attribute value equals the selected one.
function updateSelectionButtons(attribute, selected) {
    document.querySelectorAll("button[" + attribute + "]").forEach(function(button) {
        var isSelected = button.getAttribute(attribute) === selected;
        button.classList.toggle("button-17-selected", isSelected);
        button.classList.toggle("button-17", !isSelected);
    });
}

function setComparisonMethod(method) {
    if (method === "Ours" || !stackedMethodOrder.hasOwnProperty(method)) {
        return;
    }
    if (comparisonMethod === method) {
        return;
    }

    comparisonMethod = method;
    updateSelectionButtons("data-method", comparisonMethod);
}

function setComparisonScene(scene) {
    if (comparisonScenes.indexOf(scene) < 0 || comparisonScene === scene) {
        return;
    }

    comparisonScene = scene;
    updateSelectionButtons("data-comparison-scene", comparisonScene);
    loadComparisonScene();
}

function setupComparison() {
    var canvas = document.getElementById("comparison_canvas");
    var stackedVideo = document.getElementById("comparison_video");
    if (!canvas || !stackedVideo) {
        return;
    }

    var ctx = canvas.getContext("2d");
    var state = {
        position: 0.5,
        leftButtonDown: false,
        strokeColor: "#FFFFFF44",
        subVidHeight: 0,
        loadedScene: null,
        isNearViewport: false
    };

    function resizeCanvas() {
        if (!stackedVideo.videoWidth || !stackedVideo.videoHeight) {
            return;
        }
        state.subVidHeight = Math.floor(stackedVideo.videoHeight / Object.keys(stackedMethodOrder).length);
        // Resizing clears the canvas, so only do it when the size actually changes between scenes
        if (canvas.width !== stackedVideo.videoWidth || canvas.height !== state.subVidHeight) {
            canvas.width = stackedVideo.videoWidth;
            canvas.height = state.subVidHeight;
        }
    }

    function updatePosition(e) {
        var bcr = canvas.getBoundingClientRect();
        var normalized = (e.clientX - bcr.left) / bcr.width;
        if (Math.abs(normalized - state.position) < 0.1) {
            state.strokeColor = "#FFFFFFAA";
        } else {
            state.strokeColor = "#FFFFFF44";
        }
        if (state.leftButtonDown && Math.abs(normalized - state.position) < 0.5) {
            state.position = clamp(normalized, 0.0, 1.0);
        }
    }

    function drawLoop() {
        if (state.isNearViewport && canvas.width > 0 && canvas.height > 0 && stackedVideo.readyState > 2 && state.subVidHeight > 0) {
            var width = canvas.width;
            var height = canvas.height;
            var splitX = clamp(width * state.position, 0.0, width);
            var rightWidth = clamp(width - splitX, 0.0, width);
            var oursRow = stackedMethodOrder.Ours;
            var otherRow = stackedMethodOrder[comparisonMethod];

            ctx.drawImage(stackedVideo, 0, oursRow * state.subVidHeight, width, height, 0, 0, width, height);
            ctx.drawImage(stackedVideo, splitX, otherRow * state.subVidHeight, rightWidth, height, splitX, 0, rightWidth, height);

            ctx.beginPath();
            ctx.moveTo(splitX, 0);
            ctx.lineTo(splitX, height);
            ctx.closePath();
            ctx.strokeStyle = state.strokeColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            var arrowPosY = height / 2;
            var arrowW = height / 70;
            var arrowL = height / 150;
            var arrowOffset = height / 150;

            ctx.beginPath();
            ctx.moveTo(splitX + arrowL + arrowOffset, arrowPosY - arrowW / 2);
            ctx.lineTo(splitX + 2 * arrowL + arrowOffset, arrowPosY);
            ctx.lineTo(splitX + arrowL + arrowOffset, arrowPosY + arrowW / 2);
            ctx.strokeStyle = state.strokeColor;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(splitX - arrowL - arrowOffset, arrowPosY - arrowW / 2);
            ctx.lineTo(splitX - 2 * arrowL - arrowOffset, arrowPosY);
            ctx.lineTo(splitX - arrowL - arrowOffset, arrowPosY + arrowW / 2);
            ctx.strokeStyle = state.strokeColor;
            ctx.stroke();

            var insetHeight = Math.max(24, height * 0.08);
            var insetPadX = Math.max(8, width * 0.015);
            var insetPadY = Math.max(8, height * 0.02);
            ctx.font = "600 " + Math.max(12, Math.floor(height * 0.04)) + "px Jost, sans-serif";
            ctx.textBaseline = "middle";

            var leftText = "Ours";
            var rightText = comparisonMethod;
            var leftInsetWidth = ctx.measureText(leftText).width + 20;
            var rightInsetWidth = ctx.measureText(rightText).width + 20;
            var rightInsetX = width - insetPadX - rightInsetWidth;

            // Clip labels to their visible side so they disappear when covered by the slider.
            if (splitX > 0) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, splitX, height);
                ctx.clip();
                ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
                ctx.fillRect(insetPadX, insetPadY, leftInsetWidth, insetHeight);
                ctx.fillStyle = "#FFFFFF";
                ctx.fillText(leftText, insetPadX + 10, insetPadY + insetHeight / 2);
                ctx.restore();
            }

            if (splitX < width) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(splitX, 0, width - splitX, height);
                ctx.clip();
                ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
                ctx.fillRect(rightInsetX, insetPadY, rightInsetWidth, insetHeight);
                ctx.fillStyle = "#FFFFFF";
                ctx.fillText(rightText, rightInsetX + 10, insetPadY + insetHeight / 2);
                ctx.restore();
            }
        }

        requestAnimationFrame(drawLoop);
    }

    function playOrPause() {
        if (state.isNearViewport) {
            // play() is rejected e.g. in background tabs; the autoplay attribute starts playback later.
            var playPromise = stackedVideo.play();
            if (playPromise !== undefined) {
                playPromise.catch(function() {});
            }
        } else {
            stackedVideo.pause();
        }
    }

    // Pointer events cover mouse, touch and pen; the capture keeps a drag going outside the canvas.
    canvas.addEventListener("pointermove", updatePosition, false);
    canvas.addEventListener("pointerdown", function(e) {
        if (e.button === 0) {
            state.leftButtonDown = true;
            canvas.setPointerCapture(e.pointerId);
            updatePosition(e);
        }
    }, false);
    canvas.addEventListener("pointerup", function(e) {
        if (e.button === 0) {
            state.leftButtonDown = false;
        }
    }, false);
    canvas.addEventListener("pointercancel", function() {
        // The browser took over the gesture, e.g. to scroll the page
        state.leftButtonDown = false;
    }, false);
    canvas.addEventListener("pointerleave", function() {
        state.leftButtonDown = false;
        state.strokeColor = "#FFFFFF44";
    }, false);

    stackedVideo.addEventListener("loadedmetadata", resizeCanvas);

    // Swaps in the selected scene's video; the last frame stays visible, dimmed, until the new one can play
    loadComparisonScene = function() {
        if (state.loadedScene === comparisonScene) {
            return;
        }
        state.loadedScene = comparisonScene;
        if (canvas.width > 0 && canvas.height > 0) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "600 " + Math.max(12, Math.floor(canvas.height * 0.05)) + "px Jost, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Loading " + comparisonScene + "...", canvas.width / 2, canvas.height / 2);
            ctx.textAlign = "start";
        }
        stackedVideo.src = getStackedSceneVideoPath(comparisonScene);
        stackedVideo.load();
        playOrPause();
    };

    // Load the video only once the comparison approaches the viewport, and pause it while it is far away
    new IntersectionObserver(function(entries) {
        state.isNearViewport = entries[entries.length - 1].isIntersecting;
        if (state.isNearViewport && state.loadedScene === null) {
            loadComparisonScene();
        } else if (state.loadedScene !== null) {
            playOrPause();
        }
    }, { rootMargin: "50% 0px" }).observe(canvas);
    requestAnimationFrame(drawLoop);
}

function initVideoComparison() {
    setupComparison();
    updateSelectionButtons("data-method", comparisonMethod);
    updateSelectionButtons("data-comparison-scene", comparisonScene);
}
