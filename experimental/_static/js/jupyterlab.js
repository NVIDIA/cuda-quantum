var launchJupyterLab = (frame_id, notebook_path = null) => {
    let token = prompt("Enter Token:");
    let url="http://cuda-quantum-1.development.dli-infra.nvidia.com:8888"; // "http://localhost:8888/doc/"
    var frame = document.getElementById(frame_id);
    var path = notebook_path == null ? "" : "tree/" + notebook_path;
    frame.src = url + path + "?token=" + token;
}
