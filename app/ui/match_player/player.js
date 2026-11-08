const SVG_NS = "http://www.w3.org/2000/svg";
const RADIUS = { "team-a": 0.38, "team-b": 0.38, ball: 0.17 }; // metres
const TRAIL_SECONDS = 1.5;
const MS_TO_KMH = 3.6;
const HALFWAY_X = 7;

function svgElement(tag, className) {
  const element = document.createElementNS(SVG_NS, tag);
  if (className) element.setAttribute("class", className);
  return element;
}

function buildDot(track) {
  const dot = svgElement("g", track.kind);
  const circle = svgElement("circle");
  circle.setAttribute("r", RADIUS[track.kind]);
  dot.append(circle);
  if (track.kind !== "ball") {
    const label = svgElement("text");
    label.textContent = track.name;
    dot.append(label);
  }
  return dot;
}

function buildRow(track) {
  const row = document.createElement("div");
  row.className = `readout-row ${track.kind}`;
  row.innerHTML = "<span class='name'></span>" + "<span></span>".repeat(5);
  row.children[0].textContent = track.name;
  return row;
}

function buildSpacing(kind) {
  const group = svgElement("g", kind);
  const line = svgElement("line");
  const label = svgElement("text");
  group.append(line, label);
  return { group, line, label };
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

// Running total per position, so the current value is a lookup and not a recount on every screen refresh
function cumulativeDistance(points) {
  const totals = [0];
  for (let i = 1; i < points.length; i++) {
    totals.push(totals[i - 1] + distance(points[i - 1], points[i]));
  }
  return totals;
}

function nearestRider(riders, ballPoint, index) {
  let nearest = riders[0];
  for (const rider of riders) {
    if (distance(rider.points[index], ballPoint) < distance(nearest.points[index], ballPoint)) nearest = rider;
  }
  return nearest;
}

// Running count of positions where team A had the rider nearest to the ball
function cumulativeTeamANearest(riders, ball) {
  const counts = [];
  let total = 0;
  ball.points.forEach((point, index) => {
    if (nearestRider(riders, point, index).kind === "team-a") total += 1;
    counts.push(total);
  });
  return counts;
}

// Streamlit calls this again on every rerun, so it must not restart the video
export default function ({ data, parentElement }) {
  const video = parentElement.querySelector("video");
  const { tracks, hz } = data;
  const riders = tracks.filter((track) => track.kind !== "ball");
  const ball = tracks.find((track) => track.kind === "ball");
  const teams = ["team-a", "team-b"].map((kind) => ({
    kind,
    members: riders.filter((rider) => rider.kind === kind),
    ...buildSpacing(kind),
  }));

  if (video.getAttribute("src") !== data.videoUrl) {
    video.setAttribute("src", data.videoUrl);
  }

  const dots = tracks.map(buildDot);
  const trails = tracks.map((track) => svgElement("polyline", track.kind));
  const rows = tracks.map(buildRow);
  const distances = tracks.map((track) => cumulativeDistance(track.points));
  const teamANearest = cumulativeTeamANearest(riders, ball);
  parentElement.querySelector(".dots").replaceChildren(...dots);
  parentElement.querySelector(".trails").replaceChildren(...trails);
  parentElement.querySelector(".readout-rows").replaceChildren(...rows);
  parentElement.querySelector(".spacing").replaceChildren(...teams.map((team) => team.group));

  const ballLink = parentElement.querySelector(".ball-link");
  const stat = (name) => parentElement.querySelector(`[data-stat="${name}"]`);
  const ballSide = stat("ball-side");
  const nearestStat = stat("nearest");
  const shareStat = stat("share");
  const spacingStat = stat("spacing");
  const shareBar = parentElement.querySelector(".share-a");

  // The video's own time is the only clock, so the map and stats can't drift, even after seeking or pausing
  function draw() {
    const position = video.currentTime * hz;
    const last = ball.points.length - 1;
    const index = Math.min(Math.floor(position), last);
    const blend = position - Math.floor(position);
    const current = new Map();

    tracks.forEach((track, i) => {
      const { points } = track;
      const a = points[index];
      const b = points[Math.min(index + 1, last)];
      const x = a[0] + (b[0] - a[0]) * blend;
      const y = a[1] + (b[1] - a[1]) * blend;
      current.set(track, [x, y]);
      dots[i].setAttribute("transform", `translate(${x} ${y})`);

      const start = Math.max(0, index - TRAIL_SECONDS * hz);
      const recent = points.slice(start, index + 1).map((p) => p.join(","));
      trails[i].setAttribute("points", [...recent, `${x},${y}`].join(" "));
    });

    const ballPoint = current.get(ball);
    const nearest = riders.reduce((best, rider) =>
      distance(current.get(rider), ballPoint) < distance(current.get(best), ballPoint) ? rider : best,
    );
    const nearestPoint = current.get(nearest);

    tracks.forEach((track, i) => {
      const [x, y] = current.get(track);
      const speed = distance(track.points[index], track.points[Math.min(index + 1, last)]) * hz * MS_TO_KMH;
      const cells = rows[i].children;
      cells[1].textContent = x.toFixed(1);
      cells[2].textContent = y.toFixed(1);
      cells[3].textContent = speed.toFixed(1);
      cells[4].textContent = distances[i][index].toFixed(1);
      cells[5].textContent = track === ball ? "-" : distance([x, y], ballPoint).toFixed(1);
      rows[i].classList.toggle("nearest", track === nearest);
    });

    ballLink.setAttribute("x1", nearestPoint[0]);
    ballLink.setAttribute("y1", nearestPoint[1]);
    ballLink.setAttribute("x2", ballPoint[0]);
    ballLink.setAttribute("y2", ballPoint[1]);

    const spacings = teams.map((team) => {
      const [first, second] = team.members.map((member) => current.get(member));
      const gap = distance(first, second);
      team.line.setAttribute("x1", first[0]);
      team.line.setAttribute("y1", first[1]);
      team.line.setAttribute("x2", second[0]);
      team.line.setAttribute("y2", second[1]);
      team.label.setAttribute("x", (first[0] + second[0]) / 2);
      team.label.setAttribute("y", (first[1] + second[1]) / 2 - 0.12);
      team.label.textContent = `${gap.toFixed(1)} m`;
      return gap;
    });

    const shareA = (100 * teamANearest[index]) / (index + 1);
    ballSide.textContent = `${ballPoint[0] < HALFWAY_X ? "Left" : "Right"} half, ${ballPoint[0].toFixed(1)} m`;
    nearestStat.textContent = `${nearest.name}, ${distance(nearestPoint, ballPoint).toFixed(1)} m`;
    shareStat.textContent = `A ${shareA.toFixed(0)}% · B ${(100 - shareA).toFixed(0)}%`;
    shareBar.style.width = `${shareA}%`;
    spacingStat.textContent = `A ${spacings[0].toFixed(1)} m · B ${spacings[1].toFixed(1)} m`;

    frame = requestAnimationFrame(draw);
  }
  let frame = requestAnimationFrame(draw);

  return () => cancelAnimationFrame(frame);
}
