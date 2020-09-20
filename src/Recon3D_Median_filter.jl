using CUDA

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")


function compile_ptx(device_n, filename)
	dev = CuDevice(device_n)
	cap = capability(dev)

	toolkit = CUDA.find_toolkit()
	nvcc = CUDA.find_cuda_binary("nvcc", toolkit)
	flags = `-arch=sm_$(cap.major)$(cap.minor)`

	run(`$nvcc $flags -ptx -o $filename.ptx $filename.cu`)
end


function median_reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
	# Transfer the events from CPU memory to GPU memory
	c_events = CUDA.CuArray(events);

	mod = CuModuleFile("src/medianFilter.ptx")
	fun = CuFunction(mod, "medianFilterKernel")

	# Initialise the sensmap array on GPU
	c_sensmap = CUDA.CuArray(sensmap);

	# Initialise the image array on GPU
	c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

	n_threads = min(max(DIMX, DIMY, DIMZ), 128)
	n_blocks = length(c_events)
	n_shmem = max(DIMX, DIMY, DIMZ)*sizeof(Slice)

	for k = 1:recon_iters
		# Initialise correction matrix memory on GPU
		c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

		@cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events, c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

		# Perform the update step
		c_image = c_image .* c_corr ./ c_sensmap

        # Median filtering
		c_tmp_image = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));
		med_threads = (4, 4, 4);
		med_blocks = (fld1(DIMX, 4), fld1(DIMY, 4), fld1(DIMZ, 4));
		cudacall(fun, (CuPtr{Float32}, CuPtr{Float32}, Int32, Int32, Int32), c_image, c_tmp_image, DIMX, DIMY, DIMZ; threads=med_threads, blocks=med_blocks)

		c_image = c_tmp_image;
	end

	# Transfer the image estimate from GPU to CPU memory
	image = Array(c_image)

	return image
end