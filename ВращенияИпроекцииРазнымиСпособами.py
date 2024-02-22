import numpy as np
import matplotlib.pyplot as plt

#Определение матриц поворота
#Вокруг оси х
def R_x(angl):
    Rx = np.array([[1,            0,             0],
                   [0, np.cos(angl), -np.sin(angl)],
                   [0, np.sin(angl),  np.cos(angl)]])
    return Rx

# вокруг оси у
def R_y(angl):
    Ry = np.array([[ np.cos(angl), 0, np.sin(angl)],
                   [ 0,            1,            0],
                   [-np.sin(angl), 0, np.cos(angl)]])
    return Ry

# вокруг оси z
def R_z(angl):
    Rz = np.array([[np.cos(angl), -np.sin(angl), 0],
                   [np.sin(angl),  np.cos(angl), 0],
                   [           0,             0, 1]])
    return Rz

#Определение произведения кватернионов
def quaternion_product(q, p):
    q_rez = np.array([q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3],
                      q[0]*p[1]+q[1]*p[0]+q[2]*p[3]-q[3]*p[2],
                      q[0]*p[2]-q[1]*p[3]+q[2]*p[0]+q[3]*p[1],
                      q[0]*p[3]+q[1]*p[2]-q[2]*p[1]+q[3]*p[0]])
    return q_rez

# Определение матрицы поворота в 3хмерном пространстве через кватернионы
# и самалетные углы(углы Тейта - Брайана)
def Q_R_Airplane_angles(alpha, beta, gamma):
    
    # кватернионы поворота по каждой из осей, нормированный на 1 
    q1 = np.array([np.cos(alpha/2), np.sin(alpha/2), 0, 0])
    q2 = np.array([np.cos(beta/2), 0, np.sin(beta/2), 0])
    q3 = np.array([np.cos(gamma/2), 0, 0, np.sin(gamma/2)])
    
    # произведение кватернионов как результирующий поворот
    q = quaternion_product(quaternion_product(q1, q2), q3)
    
    # матрица направляющих косинусов через коэфициенты кватерниона
    QR =  np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
                    [2*(q[0]*q[3] + q[1]*q[2]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])],
                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])
    return QR

# Определение матрицы поворота в 3хмерном пространстве через кватернионы
# и углы Эйлера
def Q_R_Euler_angles(psi, teta, phi):
    
    # кватернионы поворота по каждой из осей, нормированный на 1 
    q1 = np.array([np.cos(psi/2), 0, 0, np.sin(psi/2)])
    q2 = np.array([np.cos(teta/2), np.sin(teta/2), 0, 0])
    q3 = np.array([np.cos(phi/2), 0, 0, np.sin(phi/2)])
    
    # произведение кватернионов как результирующий поворот
    q = quaternion_product(quaternion_product(q1, q2), q3)
    
    # матрица направляющих косинусов через коэфициенты кватерниона
    QR =  np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
                    [2*(q[0]*q[3] + q[1]*q[2]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])],
                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])
    return QR

# Функция расчета проекций и углов к плоскостям xy и zy, после поворота, вектора 
def spherical_coord(angl1, type_of_rotation_angles, operator_type, sequence_of_rotation="default"):
    
    # определение орт
    X = np.array([1,0, 0])
    Y = np.array([0,1, 0])
    Z = np.array([0,0, 1])
    # вектор направленный по оси Х
    V_0 = X
    
    # выбор соответствующего преобразования    
    if (type_of_rotation_angles == "euler"):
        if (operator_type == "quaternions"):
            V = Q_R_Euler_angles(angl1[0], angl1[1], angl1[2]).dot(V_0)
            
        elif(operator_type == "rotation_matrices"):
            
            if (sequence_of_rotation =="default"):
                return "error - the sequence of turns is not specified"
            
            elif(sequence_of_rotation =="XZX"):
                V = R_x(angl1[0]).dot(R_z(angl1[1]).dot(R_x(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="XYX"):
                V = R_x(angl1[0]).dot(R_y(angl1[1]).dot(R_x(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="YXY"):
                V = R_y(angl1[0]).dot(R_x(angl1[1]).dot(R_y(angl1[2]).dot(V_0)))
            
            elif(sequence_of_rotation =="YZY"):
                V = R_y(angl1[0]).dot(R_z(angl1[1]).dot(R_y(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="ZYZ"):
                V = R_z(angl1[0]).dot(R_y(angl1[1]).dot(R_z(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="ZXZ"):
                V = R_z(angl1[0]).dot(R_x(angl1[1]).dot(R_z(angl1[2]).dot(V_0)))
                
            else:
                return "error - the sequence of turns is incorrect"
            
        else:
            return "error - wrong operation type"
        
    elif (type_of_rotation_angles == "tate_brian"):
        if (operator_type == "quaternions"):
            V = Q_R_Euler_angles(angl1[0], angl1[1], angl1[2]).dot(V_0)
            
        elif(operator_type == "rotation_matrices"):
            if (sequence_of_rotation =="default"):
                return "error - the sequence of turns is not specified"
            
            elif(sequence_of_rotation =="XZY"):
                V = R_x(angl1[0]).dot(R_z(angl1[1]).dot(R_y(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="XYZ"):
                V = R_x(angl1[0]).dot(R_y(angl1[1]).dot(R_z(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="YXZ"):
                V = R_y(angl1[0]).dot(R_x(angl1[1]).dot(R_z(angl1[2]).dot(V_0)))
            
            elif(sequence_of_rotation =="YZX"):
                V = R_y(angl1[0]).dot(R_z(angl1[1]).dot(R_x(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="ZYX"):
                V = R_z(angl1[0]).dot(R_y(angl1[1]).dot(R_x(angl1[2]).dot(V_0)))
                
            elif(sequence_of_rotation =="ZXY"):
                V = R_z(angl1[0]).dot(R_x(angl1[1]).dot(R_y(angl1[2]).dot(V_0)))
                
            else:
                return "error - the sequence of turns is incorrect"
                
        else:
            return "error - wrong operation type"   
        
    else:
        return "error - incorrect angle type selected"
    
    
    # Углы через проекции на соответствующие оси
    V_xy = V[:2]
    V_yz = V[1:]
    xy_angl = np.arccos((V_xy[0]*X[0]+V_xy[1]*X[1])/(np.sqrt(V_xy[0]**2+V_xy[1]**2)*np.sqrt(X[0]**2+X[1]**2)))
    zy_angl = np.arccos((V_yz[0]*Z[1]+V_yz[1]*Z[2])/(np.sqrt(V_yz[0]**2+V_yz[1]**2)*np.sqrt(Z[1]**2+Z[2]**2)))
    
    # Углы через сферическую параметризацию
    #zy
    tet = np.arctan(np.sqrt(V[0]**2 + V[1]**2)/V[2])
    #xy
    phi = np.arctan(V[1]/V[0])
    
    projection_angl = np.array([xy_angl, zy_angl])
    
    spherical_angl = np.array([phi, tet])
    
    return projection_angl, spherical_angl, V

# углы в соответствующей системе
alpha = np.pi/3
beta = np.pi/4
gamma = np.pi/8
ang = np.array([alpha, beta, gamma])

# Промежуточный вызов функции, вычисление позиции и углов
# position2 = spherical_coord(ang, type_of_rotation_angles = "tate_brian", operator_type = "quaternions")
position = spherical_coord(ang, type_of_rotation_angles = "tate_brian",
                           operator_type = "rotation_matrices",
                           sequence_of_rotation="XZY")
#координаты после поворота
V = position[2]

#два угла 
angl_projection = position[1]

# визуализация повернутого вектора
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Рисуем оси координат
ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, label=r'орта X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, label=r'орта Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, label=r'орта Z')
# Рисуем вектор
ax.quiver(0, 0, 0, V[0], V[1], V[2], color='m', arrow_length_ratio=0.1, label=r'Повернутый вектор')
# граници осей
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.legend()

plt.show()

#углы в радианах
print(angl_projection)
