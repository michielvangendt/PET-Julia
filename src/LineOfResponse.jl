struct Point
	x::Float32
	y::Float32
	z::Float32
end

@enum Plane begin
	undef_plane
    x_plane
    y_plane
	z_plane
end

struct LOR
	P1::Point
	P2::Point
	timeDiff::Float32
end

struct Event
	lor::LOR
	mainPlane::Plane
end

struct Opt_Event
	x1::Float32
	x2::Float32
	k_y::Float32
	k_z::Float32
	y_min_base::Float32
	z_min_base::Float32
	dconv::Float32
	mainPlane::Plane
end