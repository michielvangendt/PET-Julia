using CUDA

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")
include("Visualisation.jl")

function gpu_recon_test()
    DIMX = 512
    DIMY = 512
    DIMZ = 512

    image = ones(Float32, DIMX, DIMY, DIMZ)
    c_image = CUDA.CuArray(image)

    events, = read_3D("lineSource.lmdT")
    c_events = CUDA.CuArray(events)

    c_corr = CUDA.CuArray(zeros(Float32, DIMX, DIMY, DIMZ))
    c_tmp_total_values = CUDA.CuArray(zeros(Float32, length(c_events)))

    @cuda blocks=length(c_events) threads=DIMX gpu_kernel(c_events, c_image, c_corr, c_tmp_total_values, DIMX, DIMY, DIMZ)

    c_image = c_image .* c_corr
    c_image = c_image ./ maximum(c_image)
	
	a = Array(c_image);
	save_image(a)

    visualise(c_image, DIMX, DIMY, DIMZ)
end
