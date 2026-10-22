"""Packet-coupled environment for persistent services on moving hosts."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from radio.packets import Packet
from services import (LifecycleAction, RequestEngine, ServiceLifecycleManager,
                      ServiceRegistry, ServiceRequest, ServiceSpec)


@dataclass
class PendingRequest:
    request: ServiceRequest
    host: str
    start_time: float
    cycles_remaining: float
    deadline: float
    local: bool = False
    ready: bool = False
    waiting_service: bool = False


class TransientHostEnv:
    """Minimal live transition loop for the frozen TransientHost model.

    SUMO/contact objects and helpers are injected. All remote service traffic
    enters the existing Mode-2 MAC. Lifecycle image progress is advanced from
    delivered packet bytes, so loading/migration cannot complete instantly.
    """
    def __init__(self, trace_loader, config, service_specs: dict[str, ServiceSpec],
                 mac, world, helpers: dict, host_memory: dict[str, int]):
        self.trace_loader = trace_loader
        self.config = dict(config)
        self.mac = mac
        self.world = world
        self.helpers = helpers
        self.registry = ServiceRegistry(service_specs, host_memory)
        self.lifecycle = ServiceLifecycleManager(self.registry)
        self.requests = RequestEngine(self.registry)
        self.step_dt = float(self.config.get("step_length_s", .01))
        self.time_s = 0.0
        self.pending_packets: list[Packet] = []
        self.pending_requests: dict[int, PendingRequest] = {}
        self.completed: list[dict] = []
        self.dropped = 0
        self.instance_hours = 0.0
        self.lifecycle_trace = []
        self.routing_log = []
        self._contact_cache = {}
        # Causal service-demand history used by lifecycle policies.  It is
        # updated only when a request is actually generated, never from a
        # future workload label.
        self.demand_ewma = 0.0
        self.demand_request_count = 0
        self.demand_opportunity_count = 0
        self.last_request_time = None
        self.interarrival_sum = 0.0
        self.interarrival_count = 0
        self.demand_region_counts = {}
        self.runtime_stats = {}

    def reset(self, seed=None):
        self.time_s = 0.0
        self.pending_packets.clear(); self.pending_requests.clear()
        self.completed.clear(); self.dropped = 0
        self.instance_hours = 0.0; self.lifecycle_trace.clear()
        self.routing_log.clear()
        self._contact_cache.clear()
        self.demand_ewma = 0.0
        self.demand_request_count = 0
        self.demand_opportunity_count = 0
        self.last_request_time = None
        self.interarrival_sum = 0.0
        self.interarrival_count = 0
        self.demand_region_counts = {}
        self.runtime_stats = {"simulated_ms": 0, "explicit_slot_steps": 0,
                              "fast_forwarded_ms": 0, "batched_opportunities": 0,
                              "geometry_queries": 0, "cached_contact_queries": 0,
                              "image_eligible_slots": 0, "image_attempts": 0,
                              "image_successes": 0}
        self.mac.reset(seed)
        self.registry.instances.clear()
        self.lifecycle.migration_bytes = 0
        self.lifecycle.cold_starts = 0

    def seed_warm_instance(self, service: str, host: str):
        instance = self.registry.instance(service, host)
        instance.state = "warm"

    def snapshot(self):
        """Capture the complete mutable state at a causal branch point.

        Mobility/trace objects and helper metadata are immutable references;
        packet queues, registry/lifecycle state, MAC reservations and RNG,
        demand-history counters, and controller-independent metrics are copied.
        Branches restored from this object therefore share the exact prefix.
        """
        return copy.deepcopy(self)

    def restore(self, snapshot):
        """Restore a snapshot in place for deterministic counterfactual replay."""
        restored = copy.deepcopy(snapshot)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)
        return self

    def apply_lifecycle(self, actions):
        """Apply slow actions and enqueue image/state packets where needed."""
        results = []
        for action in actions:
            result = self.lifecycle.apply(action.action, action.service,
                                          action.target, action.source)
            results.append(result)
            if not result.accepted or result.bytes_started <= 0:
                continue
            if not action.source:
                raise ValueError("packet-backed lifecycle actions require a source host")
            packet = Packet(
                source=action.source, destination=action.target,
                job_id=-1, subtask_id=0, direction="UL",
                payload_size=result.bytes_started / 1e6,
                generation_time=self.time_s,
                deadline=float("inf"), priority=0,
                flow_id=f"lifecycle:{action.service}:{action.target}",
                traffic_class="service_image")
            self.pending_packets.append(packet)
        return results

    def add_request(self, vehicle: str, service: str):
        request = self.requests.create(vehicle, service, self.time_s)
        if self.last_request_time is not None:
            self.interarrival_sum += self.time_s - self.last_request_time
            self.interarrival_count += 1
        self.last_request_time = self.time_s
        self.demand_request_count += 1
        region = self.world.zone_id(vehicle, self.time_s)
        if region is None:
            region = ("unknown",)
        self.demand_region_counts[region] = self.demand_region_counts.get(region, 0) + 1
        self.demand_ewma = 0.8 * self.demand_ewma + 0.2
        return request

    def feasible_helpers(self, time_s, vehicle):
        """Cached contact lookup for immutable mobility during one replay."""
        key = (round(float(time_s), 3), str(vehicle))
        if key not in self._contact_cache:
            self.runtime_stats["geometry_queries"] += 1
            self._contact_cache[key] = tuple(
                self.world.get_feasible_helpers(time_s, vehicle, self.helpers))
        else:
            self.runtime_stats["cached_contact_queries"] += 1
        return self._contact_cache[key]

    def fast_forward(self, target_time):
        """Jump across an interval with no packets or pending requests."""
        target_time = float(target_time)
        if target_time <= self.time_s or self.pending_packets or self.pending_requests:
            return False
        warm_count = sum(1 for instance in self.registry.instances.values()
                         if instance.state == "warm")
        self.instance_hours += warm_count * (target_time - self.time_s) / 3600.
        self.runtime_stats["fast_forwarded_ms"] += int(round((target_time - self.time_s) * 1000))
        self.time_s = round(target_time, 10)
        return True

    def route_request(self, request: ServiceRequest, host):
        reachable = self.feasible_helpers(self.time_s, request.vehicle)
        cold_source = None
        if isinstance(host, (tuple, list)):
            host, cold_source = host
        result = self.requests.route(request, host, reachable) if host is None or host in self.registry.warm_hosts(request.service) else {"status": "remote", "host": host, "cold": True}
        spec = self.registry.specs[request.service]
        deadline = request.generation_time + spec.deadline_s
        self.routing_log.append({"request_id": request.request_id,
                                 "service": request.service,
                                 "vehicle": request.vehicle,
                                 "selected": host,
                                 "reachable": list(reachable),
                                 "warm_hosts": list(self.registry.warm_hosts(request.service))})
        if result["status"] == "local":
            # Local CPU is a host, not a universal bypass.  The service must
            # already be resident and warm on the requester.
            local_instance = self.registry.instance(request.service, request.vehicle)
            if local_instance.state != "warm":
                self.dropped += 1
                return {"status": "dropped", "request_id": request.request_id,
                        "reason": "local service instance is not warm"}
            helper = self.helpers.get(request.vehicle)
            if helper is None:
                self.dropped += 1
                return {"status": "dropped", "request_id": request.request_id}
            self.pending_requests[request.request_id] = PendingRequest(
                request, request.vehicle, self.time_s,
                spec.cycles_per_request, deadline, local=True, ready=True)
        else:
            waiting_service = host not in self.registry.warm_hosts(request.service)
            if waiting_service:
                instance_state = self.registry.instance(request.service, host).state
                if instance_state in {"loading", "migrating"}:
                    # A lifecycle command is already in flight; hold the
                    # request rather than issuing a duplicate or dropping it.
                    cold_source = None
                elif not cold_source:
                    self.dropped += 1
                    return {"status": "dropped", "request_id": request.request_id}
                if cold_source:
                    result = self.lifecycle.apply("replicate", request.service, host, cold_source)
                    if not result.accepted:
                        self.dropped += 1
                        return {"status": "dropped", "request_id": request.request_id}
                    self.pending_packets.append(Packet(
                        source=cold_source, destination=host, job_id=-1,
                        subtask_id=0, direction="UL",
                        payload_size=result.bytes_started / 1e6,
                        generation_time=self.time_s, deadline=deadline,
                        priority=0, flow_id=f"lifecycle:{request.service}:{host}",
                        traffic_class="service_image"))
            self.pending_requests[request.request_id] = PendingRequest(
                request, host, self.time_s, spec.cycles_per_request, deadline)
            if not waiting_service:
                self._enqueue_request_packet(request, host, deadline)
            else:
                self.pending_requests[request.request_id].waiting_service = True
        return result

    def _enqueue_request_packet(self, request, host, deadline):
        spec = self.registry.specs[request.service]
        self.pending_packets.append(Packet(
            source=request.vehicle, destination=host,
            job_id=request.request_id, subtask_id=0, direction="UL",
            payload_size=spec.input_bytes / 1e6,
            generation_time=self.time_s, deadline=deadline,
            flow_id=f"request:{request.request_id}", traffic_class="service_request"))

    def _deliver(self, packet):
        if packet.traffic_class == "service_image":
            service = packet.flow_id.split(":")[1]
            host = packet.destination
            self.lifecycle.advance({host: int(packet.payload_size * 1e6)}, 0.0)
            return
        if packet.traffic_class == "service_request":
            pending = self.pending_requests.get(packet.job_id)
            if pending:
                pending.ready = True
            return
        if packet.traffic_class == "service_response":
            pending = self.pending_requests.pop(packet.job_id, None)
            if pending:
                self.completed.append({"request_id": packet.job_id,
                                       "success": 1,
                                       "latency_s": self.time_s - pending.start_time,
                                       "host": pending.host})
                self.lifecycle.mark_served(pending.request.service, pending.host)

    def step(self, requests=()):
        for request, host in requests:
            self.route_request(request, host)
        slots = max(1, int(self.step_dt / .001))
        for slot in range(slots):
            self.runtime_stats["explicit_slot_steps"] += 1
            self.mac.advance_slot()
            current = self.time_s
            current = self.time_s + slot * .001
            image_bytes = {}
            for packet in list(self.pending_packets):
                if current > packet.deadline:
                    self.pending_packets.remove(packet); continue
                # Every service image, request, and response transfer must
                # satisfy the same causal contact condition as ordinary
                # sidelink traffic; no controller receives a permanent link.
                feasible = self.feasible_helpers(current, packet.source)
                if packet.destination not in feasible:
                    continue
                # Lifecycle images are progressive fluid transfers.  Each
                # successful slot can deliver at most the shared Mode-2
                # capacity; contact loss leaves the remaining bytes queued.
                image_chunk = None
                if packet.traffic_class == "service_image":
                    self.runtime_stats["image_eligible_slots"] += 1
                    if not hasattr(packet, "remaining_bits"):
                        packet.remaining_bits = packet.payload_size * 8_000_000
                    image_chunk = min(packet.remaining_bits,
                                      self.mac.slot_capacity_bits)
                    packet.payload_size = image_chunk / 8_000_000
                if self.mac.attempt_transmission(packet, packet.source,
                                                  packet.destination, None, current):
                    if packet.traffic_class == "service_image":
                        self.runtime_stats["image_attempts"] += 1
                        self.runtime_stats["image_successes"] += 1
                        packet.remaining_bits -= image_chunk
                        delivered = int(image_chunk / 8)
                        image_bytes[packet.destination] = image_bytes.get(packet.destination, 0) + delivered
                        if packet.remaining_bits <= 0:
                            self.pending_packets.remove(packet)
                    else:
                        self.pending_packets.remove(packet)
                        self._deliver(packet)
                elif packet.traffic_class == "service_image":
                    self.runtime_stats["image_attempts"] += 1
            # Service compute and response generation.
            for rid, pending in list(self.pending_requests.items()):
                if pending.local:
                    rate = self.helpers[pending.host].available_cpu_rate()
                else:
                    rate = self.helpers[pending.host].available_cpu_rate()
                if pending.waiting_service or not pending.ready or pending.cycles_remaining == float("inf"):
                    continue
                pending.cycles_remaining -= rate * .001
                if pending.cycles_remaining <= 0 and not pending.local:
                    spec = self.registry.specs[pending.request.service]
                    self.pending_packets.append(Packet(
                        source=pending.host, destination=pending.request.vehicle,
                        job_id=rid, subtask_id=0, direction="DL",
                        payload_size=spec.output_bytes / 1e6,
                        generation_time=current, deadline=pending.deadline,
                        flow_id=f"response:{rid}", traffic_class="service_response"))
                    pending.cycles_remaining = float("inf")
                elif pending.cycles_remaining <= 0 and pending.local:
                    self.pending_requests.pop(rid)
                    self.completed.append({"request_id": rid, "success": 1,
                                           "latency_s": current - pending.start_time,
                                           "host": pending.host})
            self.lifecycle.advance(image_bytes, .001)
            for pending in self.pending_requests.values():
                if pending.waiting_service:
                    instance = self.registry.instance(pending.request.service, pending.host)
                    if instance.state == "warm":
                        pending.waiting_service = False
                        self._enqueue_request_packet(pending.request, pending.host, pending.deadline)
            warm_count = sum(1 for instance in self.registry.instances.values()
                             if instance.state == "warm")
            self.instance_hours += warm_count * .001 / 3600.
        # Persist one compact lifecycle snapshot per environment decision
        # epoch, rather than duplicating the full registry at every 1-ms slot.
        self.lifecycle_trace.append({"time": self.time_s,
                                     "instances": self.registry.snapshot()})
        self.time_s = round(self.time_s + self.step_dt, 10)
        self.runtime_stats["simulated_ms"] += int(round(self.step_dt * 1000))
        for rid, pending in list(self.pending_requests.items()):
            if self.time_s >= pending.deadline:
                self.pending_requests.pop(rid)
                self.dropped += 1
