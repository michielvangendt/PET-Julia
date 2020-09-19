using CUDA

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")
include("Visualisation.jl")

function gpu_recon_test()
    DIMX = 512
    DIMY = 512
    DIMZ = 512

    c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

    events, = read_3D("Triple_line_source.lmdT");
    c_events = CUDA.CuArray(events);

    c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));
	
	n_threads = min(max(DIMX, DIMY, DIMZ), 128)
	n_blocks = length(c_events)
	n_shmem = max(DIMX, DIMY, DIMZ)*sizeof(Slice)

    CUDA.@time @cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events, c_image, c_corr, DIMX, DIMY, DIMZ)

    c_image = c_image .* c_corr
    c_image = c_image ./ maximum(c_image)
	
	a = Array(c_image);
	save_image(a)

    visualise(c_image, DIMX, DIMY, DIMZ)
end
