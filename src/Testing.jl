using CUDA
using BenchmarkTools

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")
include("Recon3D_Attenuation.jl")
include("Recon3D_TOF.jl")
include("Recon3D_Median_filter.jl")
include("Visualisation.jl")

benchmarking = true

function gpu_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 4

    events, = read_3D("Prostate_Att.lmdT");

	CUDA.@time begin
		c_image = reconstruct3D(events, DIMX, DIMY, DIMZ, recon_iters)
	end

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = reconstruct3D($events, $DIMX, $DIMY, $DIMZ, $recon_iters)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

    visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end

function gpu_sensmap_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 4

	sensmap = read_sensmap("Data/Sensmap/sensmap_133_86_580_1000.raw", DIMX, DIMY, DIMZ);
    events, = read_3D("Prostate_Att.lmdT");

	CUDA.@time begin
	    c_image = reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
	end;

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = reconstruct3D($events, $sensmap, $DIMX, $DIMY, $DIMZ, $recon_iters)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

	visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end

function gpu_osem_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 2
	number_of_subsets = 2

	sensmap = read_sensmap("Data/Sensmap/sensmap_133_86_580_1000.raw", DIMX, DIMY, DIMZ);
    events, = read_3D("Prostate_Att.lmdT");

	CUDA.@time begin
	    c_image = OS_reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters, number_of_subsets)
	end;

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = OS_reconstruct3D($events, $sensmap, $DIMX, $DIMY, $DIMZ, $recon_iters, $number_of_subsets)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

	visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end

function gpu_attmap_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 4

	sensmap = read_sensmap("Data/Sensmap/sensmap_133_86_580_1000.raw", DIMX, DIMY, DIMZ);
	attmap = read_attmap("Data/Attenuation/attmap.raw", DIMX, DIMY, DIMZ);
    events, = read_3D("Prostate_Att.lmdT");

	CUDA.@time begin
	    c_image = att_reconstruct3D(events, sensmap, attmap, DIMX, DIMY, DIMZ, recon_iters)
	end;

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = att_reconstruct3D($events, $sensmap, $attmap, $DIMX, $DIMY, $DIMZ, $recon_iters)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

    visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end

function gpu_tof_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 4

	sensmap = read_sensmap("Data/Sensmap/sensmap_133_86_580_1000.raw", DIMX, DIMY, 580);
    events, = read_3D("Prostate_Att.lmdT");

	c_tof_kernel = CUDA.CuArray(create_tof_kernel());
	cc_tof_kernel = cudaconvert(c_tof_kernel)

	TOF_events = [TOF_event(event, event.lor.timeDiff, cc_tof_kernel) for event in events ];

	CUDA.@time begin
	    c_image = tof_reconstruct3D(TOF_events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
	end;

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = tof_reconstruct3D($TOF_events, $sensmap, $DIMX, $DIMY, $DIMZ, $recon_iters)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

    visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end

function gpu_median_recon_test()
	DIMX = 133
	DIMY = 86
	DIMZ = 580
	recon_iters = 4

	sensmap = read_sensmap("Data/Sensmap/sensmap_133_86_580_1000.raw", DIMX, DIMY, DIMZ);
    events, = read_3D("Prostate_Att.lmdT");

	compile_ptx(0, "src/medianFilter.cu")

	CUDA.@time begin
	    c_image = median_reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
	end;

	if benchmarking
		bench = @benchmark begin
			CUDA.@sync begin
				c_image = median_reconstruct3D($events, $sensmap, $DIMX, $DIMY, $DIMZ, $recon_iters)
			end
		end
		bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds
		print("\n\nTotal GPU time $bench_time seconds \n\n\n");
	end

	visualise(c_image, DIMX, DIMY, DIMZ)
	save_image(c_image, DIMX, DIMY, DIMZ)
end