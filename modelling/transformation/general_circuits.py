import cirq
from modelling.transformation.entangle import entangler
from modelling.transformation.rotation import layerX, layerY, layerZ

class PQCs:
    def __init__(self, bits, gen_params, pqc=1):
        self.choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert pqc in self.choices
        self.bits=bits
        self.gen_params = gen_params
        self.pqc=pqc
        self.circuit = self.get_pqc()
    def get_pqc(self):
        if self.pqc == 1:
            return self.__pqc_1(self.gen_params)
        if self.pqc == 2:
            return self.__pqc_2(self.gen_params)
        if self.pqc == 3:
            return self.__pqc_3(self.gen_params)
        if self.pqc == 4:
            return self.__pqc_4(self.gen_params)
        if self.pqc == 5:
            return self.__pqc_5(self.gen_params)
        if self.pqc == 6:
            return self.__pqc_6(self.gen_params)
        if self.pqc == 7:
            return self.__pqc_7(self.gen_params)
        if self.pqc == 8:
            return self.__pqc_8(self.gen_params)
        if self.pqc == 9:
            return self.__pqc_9(self.gen_params)
        if self.pqc == 10:
            return self.__pqc_10(self.gen_params)
        if self.pqc == 11:
            return self.__pqc_11(self.gen_params)
        if self.pqc == 12:
            return self.__pqc_12(self.gen_params)
        if self.pqc == 13:
            return self.__pqc_13(self.gen_params)
        if self.pqc == 14:
            return self.__pqc_14(self.gen_params)
        if self.pqc == 15:
            return self.__pqc_15(self.gen_params)
        if self.pqc == 16:
            return self.__pqc_16(self.gen_params)
        if self.pqc == 17:
            return self.__pqc_17(self.gen_params)
        if self.pqc == 18:
            return self.__pqc_18(self.gen_params)
        if self.pqc == 19:
            return self.__pqc_19(self.gen_params)
        
    def __pqc_1(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        return circuit

    def __pqc_2(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)-1):
            circuit.append(cirq.CNOT(target=self.bits[i], control=self.bits[i+1]))    
        return circuit

    def __pqc_3(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)-1):
            circuit.append(cirq.rz(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1]))    
        return circuit

    def __pqc_4(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)-1):
            circuit.append(cirq.rx(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1]))    
        return circuit

    def __pqc_5(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)):
            for j in range(len(self.bits)-1):
                target = self.bits[(i+j+1)%len(self.bits)]
                control = self.bits[i]
                circuit.append(cirq.rz(gen_params()).on(target).controlled_by(control))  
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))  
        return circuit
    
    def __pqc_6(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)):
            for j in range(len(self.bits)-1):
                target = self.bits[(i+j+1)%len(self.bits)]
                control = self.bits[i]
                circuit.append(cirq.rx(gen_params()).on(target).controlled_by(control))  
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))  
        return circuit

    def __pqc_7(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))

        for i in range(0, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.rz(gen_params()).on(qubit1).controlled_by(qubit2))

        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))  

        for i in range(1, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.rz(gen_params()).on(qubit1).controlled_by(qubit2))

        return circuit
    
    def __pqc_8(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))

        for i in range(0, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.rx(gen_params()).on(qubit1).controlled_by(qubit2))

        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))  

        for i in range(1, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.rx(gen_params()).on(qubit1).controlled_by(qubit2))

        return circuit
    
    def __pqc_9(self, gen_params):
        circuit = cirq.Circuit()
        
        for i in range(len(self.bits)):
            circuit.append(cirq.H(self.bits[i]))
        for i in range(len(self.bits)-1):
            circuit.append(cirq.CZ(self.bits[i], self.bits[i+1]))

        circuit.append(layerX(self.bits, gen_params=gen_params))
        return circuit
    
    def __pqc_10(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerY(self.bits, gen_params=gen_params))
        for i in range(len(self.bits)):
            circuit.append(cirq.CZ(self.bits[i], self.bits[(i+1)%len(self.bits)]))

        circuit.append(layerY(self.bits, gen_params=gen_params))
        return circuit

    def __pqc_11(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerY(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0, len(self.bits)-1, 2):
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.CNOT(control=qubit2, target=qubit1))
        for i in range(1, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.ry(gen_params()).on(qubit1))
            circuit.append(cirq.rz(gen_params()).on(qubit1))
            circuit.append(cirq.ry(gen_params()).on(qubit2))
            circuit.append(cirq.rz(gen_params()).on(qubit2))
            circuit.append((cirq.CNOT(control=qubit2, target=qubit1)))

        return circuit

    def __pqc_12(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerY(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0, len(self.bits)-1, 2):
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.CZ(qubit1,qubit2))
        for i in range(1, len(self.bits), 2):
            if i >= (len(self.bits)-1):
                break
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1)]    
            circuit.append(cirq.ry(gen_params()).on(qubit1))
            circuit.append(cirq.rz(gen_params()).on(qubit1))
            circuit.append(cirq.ry(gen_params()).on(qubit2))
            circuit.append(cirq.rz(gen_params()).on(qubit2))
            circuit.append((cirq.CZ(qubit1,qubit2)))

        return circuit
    
    def __pqc_13(self, gen_params):
        n = len(self.bits)

        circuit1 = cirq.Circuit()
        for qubit in self.bits:
            circuit1.append(cirq.ry(gen_params()).on(qubit))

        circuit2 = cirq.Circuit()
        for i in range(n-1, -1, -1):
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1) % n]
            circuit2.append(cirq.rz(gen_params()).on(qubit2).controlled_by(qubit1))

        circuit3 = cirq.Circuit()
        for qubit in self.bits:
            circuit3.append(cirq.ry(gen_params()).on(qubit))

        circuit4 = cirq.Circuit()
        for i in range(n):
            j = i + n//2
            qubit1 = self.bits[j % n]
            qubit2 = self.bits[(j+1) % n]
            circuit4.append(cirq.rz(gen_params()).on(qubit1).controlled_by(qubit2))
            
        circuit = cirq.Circuit()
        circuit.append(circuit1)
        circuit.append(circuit2)
        circuit.append(circuit3)
        circuit.append(circuit4)
        return circuit
    
    
    def __pqc_14(self, gen_params):
        n = len(self.bits)

        circuit1 = cirq.Circuit()
        for qubit in self.bits:
            circuit1.append(cirq.ry(gen_params()).on(qubit))

        circuit2 = cirq.Circuit()
        for i in range(n-1, -1, -1):
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1) % n]
            circuit2.append(cirq.rx(gen_params()).on(qubit2).controlled_by(qubit1))

        circuit3 = cirq.Circuit()
        for qubit in self.bits:
            circuit3.append(cirq.ry(gen_params()).on(qubit))

        circuit4 = cirq.Circuit()
        for i in range(n):
            j = i + n//2
            qubit1 = self.bits[j % n]
            qubit2 = self.bits[(j+1) % n]
            circuit4.append(cirq.rx(gen_params()).on(qubit1).controlled_by(qubit2))
            
        circuit = cirq.Circuit()
        circuit.append(circuit1)
        circuit.append(circuit2)
        circuit.append(circuit3)
        circuit.append(circuit4)
        return circuit
    
    def __pqc_15(self, gen_params):
        n = len(self.bits)

        circuit1 = cirq.Circuit()
        for qubit in self.bits:
            circuit1.append(cirq.ry(gen_params()).on(qubit))

        circuit2 = cirq.Circuit()
        for i in range(n-1, -1, -1):
            qubit1 = self.bits[i]
            qubit2 = self.bits[(i+1) % n]
            circuit2.append(cirq.CNOT(target=qubit2, control=qubit1))

        circuit3 = cirq.Circuit()
        for qubit in self.bits:
            circuit3.append(cirq.ry(gen_params()).on(qubit))

        circuit4 = cirq.Circuit()
        for i in range(n):
            j = i + n//2
            qubit1 = self.bits[j % n]
            qubit2 = self.bits[(j+1) % n]
            circuit4.append(cirq.CNOT(target=qubit1, control=qubit2))
            
        circuit = cirq.Circuit()
        circuit.append(circuit1)
        circuit.append(circuit2)
        circuit.append(circuit3)
        circuit.append(circuit4)
        return circuit
    
    def __pqc_16(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0,len(self.bits)-1, 2):
            circuit.append(cirq.rz(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1])) 
        for i in range(1,len(self.bits)-1, 2):
            circuit.append(cirq.rz(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1]))     
        return circuit

    def __pqc_17(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0,len(self.bits)-1, 2):
            circuit.append(cirq.rx(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1])) 
        for i in range(1,len(self.bits)-1, 2):
            circuit.append(cirq.rx(gen_params()).on(self.bits[i]).controlled_by(self.bits[i+1]))     
        return circuit

    def __pqc_18(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0,len(self.bits)):

            circuit.append(cirq.rz(gen_params()).on(self.bits[i]).controlled_by(self.bits[(i + len(self.bits) - 1) % len(self.bits)])) 

        return circuit

    def __pqc_19(self, gen_params):
        circuit = cirq.Circuit()
        
        circuit.append(layerX(self.bits, gen_params=gen_params))
        circuit.append(layerZ(self.bits, gen_params=gen_params))
        for i in range(0,len(self.bits)):

            circuit.append(cirq.rx(gen_params()).on(self.bits[i]).controlled_by(self.bits[(i + len(self.bits) - 1) % len(self.bits)])) 

        return circuit
     